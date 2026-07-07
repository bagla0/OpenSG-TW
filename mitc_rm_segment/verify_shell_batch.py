"""verify_shell_batch.py -- certify the BATCHED assembly against the original
per-element loop (replicated here from the pre-optimization code, using the
retained scalar quad_ops_indep/_mitc_shear_indep), on real meshes:

  a) tapered square thin iso segment (full shear, kg_e)
  b) tapered circle thin m45 segment (curved, kg_e)
  c) wrapped ring strip with dof_map, all four shear schemes
  d) constraint operators (elem) + build_C_Psi_segment6

Then the end-to-end pipeline result + timing.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)

import segment_indep as si
from segment_indep import (quad_ops_indep, _mitc_shear_indep, NDOF6,
                           assemble_segment_indep, assemble_constraint,
                           build_C_Psi_segment6, _d_scale)
from segment_element import _bilinear, compute_k22, compute_kg
from solve_segment_jax import _material_by_section
from boundary_from_yaml import extract
import io, contextlib


def assemble_loop(nodes, quads, subdom, e3s, D_by, G_by, k22_e, cross, ax,
                  kg_e=None, pen=None, pen_beta=0.1, dof_map=None, shear="full"):
    """VERBATIM pre-optimization per-element assembly."""
    if pen is None:
        pen = pen_beta * _d_scale(D_by)
    if dof_map is None:
        dof_map = np.arange(len(nodes))
    Nn = int(np.max(dof_map)) + 1; ndof = NDOF6 * Nn
    Dhh = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 4)); Dee = np.zeros((4, 4))
    Dhl = np.zeros((ndof, ndof)); Dll = np.zeros((ndof, ndof)); Dle = np.zeros((ndof, 4))
    gpv = 1.0 / np.sqrt(3.0)
    gp = [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]
    for q, quad in enumerate(quads):
        X = nodes[quad]; k22 = float(k22_e[q]); kg = float(kg_e[q]) if kg_e is not None else 0.0
        D = D_by[int(subdom[q])]; G = G_by[int(subdom[q])]
        g = np.concatenate([[NDOF6 * int(dof_map[nd]) + cc for cc in range(NDOF6)] for nd in quad])
        gij = (g[:, None], g[None, :])
        for (xi, eta) in gp:
            BDe, BDh, BDl, BGe, BGh, BGl, DRe, DRh, DRl, dA = quad_ops_indep(
                X, e3s[q], xi, eta, k22, cross, ax, kg)
            BGt = BGh if shear == "full" else _mitc_shear_indep(
                X, e3s[q], xi, eta, k22, cross, ax, kg, scheme=shear)
            w = dA
            DRh2 = DRh[:, None]; DRl2 = DRl[:, None]
            np.add.at(Dhh, gij, (BDh.T @ D @ BDh + BGt.T @ G @ BGt + pen * (DRh2 @ DRh2.T)) * w)
            np.add.at(Dhe, g, (BDh.T @ D @ BDe + BGt.T @ G @ BGe + pen * (DRh2 @ DRe[None, :])).squeeze() * w)
            Dee += (BDe.T @ D @ BDe + BGe.T @ G @ BGe + pen * np.outer(DRe, DRe)) * w
            np.add.at(Dhl, gij, (BDh.T @ D @ BDl + BGt.T @ G @ BGl + pen * (DRh2 @ DRl2.T)) * w)
            np.add.at(Dll, gij, (BDl.T @ D @ BDl + BGl.T @ G @ BGl + pen * (DRl2 @ DRl2.T)) * w)
            np.add.at(Dle, g, (BDl.T @ D @ BDe + BGl.T @ G @ BGe + pen * (DRl2 @ DRe[None, :])).squeeze() * w)
    return Dhh, Dhe, Dee, Dhl, Dll, Dle


def constraint_loop(nodes, quads, subdom, e3s, k22_e, cross, ax, kg_e=None, dof_map=None):
    """VERBATIM pre-optimization elem-lambda constraint assembly."""
    Nn = len(nodes)
    if dof_map is None:
        dof_map = np.arange(Nn)
    Nd = int(np.max(dof_map)) + 1; M = NDOF6 * Nd
    P = len(quads)
    G = np.zeros((P, M)); Gl = np.zeros((P, M)); Ge = np.zeros((P, 4))
    gpv = 1.0 / np.sqrt(3.0)
    gp = [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]
    for q, quad in enumerate(quads):
        X = nodes[quad]; k22 = float(k22_e[q]); kg = float(kg_e[q]) if kg_e is not None else 0.0
        gloc = np.array([NDOF6 * int(dof_map[nd]) + c for nd in quad for c in range(NDOF6)])
        for (xi, eta) in gp:
            _, _, _, _, _, _, DRe, DRh, DRl, dA = quad_ops_indep(X, e3s[q], xi, eta, k22, cross, ax, kg)
            np.add.at(G[q], gloc, DRh * dA)
            np.add.at(Gl[q], gloc, DRl * dA)
            Ge[q] += DRe * dA
    return G, Gl, Ge


def cpsi_loop(nodes, quads, cross):
    Nn = len(nodes); ndof = NDOF6 * Nn
    C = np.zeros((4, ndof)); Psi = np.zeros((ndof, 4))
    gpv = 1.0 / np.sqrt(3.0)
    qp = [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]
    for quad in quads:
        X = nodes[quad]
        for (xi, eta) in qp:
            Nn_, dNx, dNe = _bilinear(xi, eta)
            dA = np.linalg.norm(np.cross(dNx @ X, dNe @ X))
            for a, nd in enumerate(quad):
                for cc in range(4):
                    C[cc, NDOF6 * nd + cc] += Nn_[a] * dA
    for nd in range(Nn):
        y2, y3 = nodes[nd, cross[0]], nodes[nd, cross[1]]
        Psi[NDOF6 * nd + 0, 0] = 1.0
        Psi[NDOF6 * nd + 1, 1] = 1.0
        Psi[NDOF6 * nd + 2, 2] = 1.0
        Psi[NDOF6 * nd + 1, 3] = -y3; Psi[NDOF6 * nd + 2, 3] = y2; Psi[NDOF6 * nd + 3, 3] = -1.0
    return C, Psi


def relmax(A, B, scale=None):
    A, B = np.asarray(A), np.asarray(B)
    den = np.abs(B).max() if scale is None else scale
    den = den if den > 0 else 1.0
    return np.abs(A - B).max() / den


def load_case(sub, tg):
    npz = os.path.join(HERE, "out", "vb_scratch", "%s_%s.npz" % (sub, tg))
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        extract(os.path.join(HERE, "out", sub, "meshes", "shell_%s.yaml" % tg), npz, plot=False)
    b = np.load(npz, allow_pickle=True)
    ax = int(b["axis"]); cross = tuple(j for j in range(3) if j != ax)
    D_by, G_by = _material_by_section(json.loads(str(b["sections"])),
                                      json.loads(str(b["materials"])), center_ref=True)
    return b, ax, cross, D_by, G_by


names = ["Dhh", "Dhe", "Dee", "Dhl", "Dll", "Dle"]
for sub, tg in (("taper_square", "thin_iso_aR070"), ("taper_study", "thin_m45_aR070")):
    b, ax, cross, D_by, G_by = load_case(sub, tg)
    nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"]); sd = np.asarray(b["seg_subdom"])
    e1s, e2s, e3s = np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]), np.asarray(b["seg_e3"])
    cents = nodes[quads].mean(1)
    k22 = compute_k22(cents, e2s, e3s, quads); kg = compute_kg(cents, e1s, e2s, e3s, quads)
    A_new = assemble_segment_indep(nodes, quads, sd, e3s, D_by, G_by, k22, cross, ax, kg_e=kg, pen=0.0)
    A_old = assemble_loop(nodes, quads, sd, e3s, D_by, G_by, k22, cross, ax, kg_e=kg, pen=0.0)
    print("== %s %s segment ==" % (sub, tg))
    for nm, X, Y in zip(names, A_new, A_old):
        print("  %-4s relmax %.3e" % (nm, relmax(X, Y)))
    Gn, Gln, Gen = assemble_constraint(nodes, quads, sd, e3s, k22, cross, ax, kg_e=kg)
    Go, Glo, Geo = constraint_loop(nodes, quads, sd, e3s, k22, cross, ax, kg_e=kg)
    print("  G    relmax %.3e | Gl %.3e | Ge %.3e"
          % (relmax(Gn, Go), relmax(Gln, Glo), relmax(Gen, Geo, scale=max(np.abs(Geo).max(), 1e-30))))
    Cn, Pn = build_C_Psi_segment6(nodes, quads, cross)
    Co, Po = cpsi_loop(nodes, quads, cross)
    print("  C    relmax %.3e | Psi %.3e" % (relmax(Cn, Co), relmax(Pn, Po)))

# ring wrapped strip (dof_map + all shear schemes)
b, ax, cross, D_by, G_by = load_case("taper_square", "thin_m45_aR070")
rx = np.asarray(b["L_x"]); rc = np.asarray(b["L_cells"])
rs = np.asarray(b["L_subdom"]); re3 = np.asarray(b["L_e3"])
kr = compute_k22(rx[rc].mean(1), np.asarray(b["L_e2"]), re3, rc)
m = len(rx)
h = float(np.mean(np.linalg.norm(rx[rc[:, 1]] - rx[rc[:, 0]], axis=1)))
ez = np.zeros(3); ez[ax] = 1.0
rn = np.vstack([rx, rx + h * ez])
dmap = np.concatenate([np.arange(m), np.arange(m)])
rquads = np.array([[a, bq, m + bq, m + a] for a, bq in rc], dtype=int)
print("== ring strip (dof_map), all shear schemes ==")
for sch in ("full", "mitc4_g23", "mitc4_wonly", "mitc4_both"):
    A_new = assemble_segment_indep(rn, rquads, rs, re3, D_by, G_by, kr, cross, ax,
                                   pen=0.0, dof_map=dmap, shear=sch)
    A_old = assemble_loop(rn, rquads, rs, re3, D_by, G_by, kr, cross, ax,
                          pen=0.0, dof_map=dmap, shear=sch)
    worst = max(relmax(X, Y) for X, Y in zip(A_new, A_old))
    print("  %-12s worst relmax %.3e" % (sch, worst))
Gn, Gln, Gen = assemble_constraint(rn, rquads, rs, re3, kr, cross, ax, dof_map=dmap)
Go, Glo, Geo = constraint_loop(rn, rquads, rs, re3, kr, cross, ax, dof_map=dmap)
print("  ring G relmax %.3e | Gl %.3e" % (relmax(Gn, Go), relmax(Gln, Glo)))

# end-to-end
import run_indep as ri
RES = os.path.join(HERE, "out", "vb_res"); os.makedirs(RES, exist_ok=True)
r = ri.shell_solve_lagrange("thin_iso_aR070", os.path.join(HERE, "out", "taper_square", "meshes"),
                            RES, return_full=True)
t0 = time.perf_counter()
r = ri.shell_solve_lagrange("thin_iso_aR070", os.path.join(HERE, "out", "taper_square", "meshes"),
                            RES, return_full=True)
t1 = time.perf_counter()
ref = np.array([8.93614749e9, 1.22412762e9, 1.22412762e9, 2.47400314e9, 4.30597032e9, 4.30597032e9])
print("== end-to-end square thin iso ==")
print("  total %.2fs (extract %.2f rings %.2f seg %.2f)"
      % (t1 - t0, r["t_extract"], r["t_rings"], r["t_seg"]))
print("  seg diag:", np.diag(r["S6"]))
print("  vs pre-optimization diag relmax: %.3e"
      % (np.abs(np.diag(r["S6"]) - ref).max() / ref.max()))
