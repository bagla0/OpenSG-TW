"""timing4.py -- wall-clock cost of the three homogenizers on ALL FOUR paper examples:
  Ex1  webbed two-cell tube (iso thin)      : stored 2-D solid yaml
  Ex2  elliptic 4-cell tube (iso thin)      : generated 2-D solid yaml
  Ex3  IEA-22 r/R=0.2 cross-section         : prismatic hex section (timing_r020 machinery)
  Ex3.1 full IEA-22 blade (8 stations, sum) : same machinery per station

Methods: RM 6-DOF shell ring (1-D contour) | OpenSG-JAX 2-D solid | OpenSG-FEniCS 2-D solid.
FEniCS route: the 2-D quad section is extruded into a thin prismatic hex segment and
compute_timo_boun is run on its LEFT face (= the 2-D cross-section) -- the same proven
path as timing_r020.  JAX is timed on the second call (compiled kernel); the one-off JIT
compile is reported separately.  Solver wall time only (mesh generation excluded).

  -> results/timing4.npz
"""
import os
import sys
import time

import numpy as np
import yaml as _yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TAPER = os.path.abspath(os.path.join(HERE, "..", "..", "taper"))
for q in (MITC, REPO, TAPER, os.path.expanduser("~/OpenSG_io")):
    sys.path.insert(0, q)

from xsec_5v6_master import load_ring, ring_6dof
from oml_ring import load_ring_ref, c6
from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block, emit_opensg_yaml
from opensg_io.hex_loft import hex_between_sections, solid_yaml_payload
from opensg_jax.fe_jax.solid_taper import (split_batches_to_tets, extract_boundary_submesh,
                                           solve_boundary, _PERM3, _PERMF, write_solid_segment_yaml)

TWP = os.path.abspath(os.path.join(HERE, ".."))
TC = os.path.join(TWP, "two_cell_tube", "data")
ELM = os.path.join(HERE, "ellipse", "meshes")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
SCR = os.path.join(OUT, "_timing4_scr"); os.makedirs(SCR, exist_ok=True)


def _row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(",", " ").split()]
    return [float(v) for v in r]


def read_solid_2d(path):
    """2-D solid yaml -> (nodes[N,2], quads[M,4] 0-based, oris[M,9], names[M], mats)."""
    d = _yaml.safe_load(open(path))
    nd = np.array([_row(n) for n in d["nodes"]], dtype=float)[:, :2]
    qs = np.array([[int(v) for v in _row(e)] for e in d["elements"]], dtype=int)
    if qs.min() == 1:
        qs = qs - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], dtype=float)
    names = np.empty(len(qs), dtype=object)
    for grp in d["sets"]["element"]:
        for lab in grp["labels"]:
            names[int(lab) - 1] = grp["name"]
    mats = []
    for mm in d["materials"]:
        e = mm.get("elastic", mm)
        mats.append({"name": mm["name"], "E": list(e["E"]), "G": list(e["G"]),
                     "nu": list(e["nu"]), "rho": float(mm.get("density", mm.get("rho", 1.0)))})
    return nd, qs, ori, list(names), mats


def t_ring(path, ref=None):
    R = load_ring_ref(path, ref) if ref else load_ring(path)
    t0 = time.time()
    C = c6(R) if ref else ring_6dof(R)
    return time.time() - t0, 6 * len(R["rx"]), C[0, 0]


def t_jax_yaml(path):
    t0 = time.time(); compute_timo_from_yaml(path, verbose=False); t1 = time.time() - t0
    t0 = time.time(); C = compute_timo_from_yaml(path, verbose=False); t2 = time.time() - t0
    d = _yaml.safe_load(open(path)); ndof = 3 * len(d["nodes"])
    return t1, t2, ndof, float(np.asarray(C)[0, 0])


def t_fenics_2d(path, tag, dz=None):
    """extrude the 2-D quad section into a 2-layer prismatic hex segment; time
    compute_timo_boun on the LEFT face (= the 2-D cross-section).  The extrusion must be
    long enough for the boundary split to isolate the end plane -- default half the
    section's larger extent per layer.  Callers must sanity-check the returned EA."""
    from opensg.mesh.segment import SolidSegmentMesh
    from opensg.core.solid import compute_timo_boun
    nd, qs, ori, names, mats = read_solid_2d(path)
    if dz is None:
        dz = 0.5 * float(max(np.ptp(nd[:, 0]), np.ptp(nd[:, 1])))
    NL = 2                                                    # axial layers
    nn = len(nd)
    nodes3 = np.vstack([np.column_stack([nd, np.full(nn, k * dz)]) for k in range(NL + 1)])
    hexes, hmats, oris = [], [], []
    for k in range(NL):
        for m, q in enumerate(qs):
            hexes.append([q[0] + k * nn, q[1] + k * nn, q[2] + k * nn, q[3] + k * nn,
                          q[0] + (k + 1) * nn, q[1] + (k + 1) * nn,
                          q[2] + (k + 1) * nn, q[3] + (k + 1) * nn])
            hmats.append(names[m]); oris.append(ori[m])
    yfe = os.path.join(SCR, "seg_%s.yaml" % tag)
    write_solid_segment_yaml(yfe, nodes3, hexes, hmats, np.asarray(oris), mats)
    cwd = os.getcwd(); os.chdir(SCR)
    try:
        sm = SolidSegmentMesh(yfe)
        mp, _den = sm.material_database
        nL = sm.left_submesh["mesh"].topology.index_map(0).size_local
        if abs(nL - len(nd)) > 0.05 * len(nd):
            raise RuntimeError("left submesh %d nodes != section %d -- boundary split "
                               "failed, timing would be invalid" % (nL, len(nd)))
        t0 = time.time(); out = compute_timo_boun(mp, sm.left_submesh); dt = time.time() - t0
        C = np.asarray(out[0])
        ndof = 3 * nL
    finally:
        os.chdir(cwd)
    return dt, ndof, C[0, 0]


def _mp_row(blade, n):
    e = _mat_block(blade, n)["elastic"]
    return [e["E"][0], e["E"][1], e["E"][2], e["G"][0], e["G"][1], e["G"][2],
            e["nu"][0], e["nu"][1], e["nu"][2]]


def iea_station(blade, r, ms=0.02, nr=4):
    tag = "r%03d" % int(round(r * 100))
    cs = build_cross_section(blade, r, mesh_size=ms)
    sp = os.path.join(SCR, "shell_%s.yaml" % tag); emit_opensg_yaml(cs, sp, fraction=0.0)
    ts, ds, _ = t_ring(sp, ref="oml")
    z1 = blade_span_z(blade, r)
    res = hex_between_sections(cs, cs, z1, z1 + 2.0, nr=nr, nsp=4, nw=3, mesh_size=ms)
    oris, hmats = solid_yaml_payload(res, cs, cs)
    mat_names = sorted(set(hmats)); nix = {n: i for i, n in enumerate(mat_names)}
    seg = dict(nodes=np.asarray(res["nodes"])[:, _PERM3],
               batches={"hex8": (np.asarray(res["hexes"]),
                                 np.array([nix[m] for m in hmats], int), np.asarray(oris)[:, _PERMF])},
               mat_param=np.array([_mp_row(blade, n) for n in mat_names]),
               nelem=len(res["hexes"]))
    web = np.array([t[0] == "web" for t in res["htag"]])
    seg = split_batches_to_tets(seg, mask=web)
    bL = extract_boundary_submesh(seg, "L")
    t0 = time.time(); solve_boundary(bL); tj1 = time.time() - t0
    t0 = time.time(); solve_boundary(bL); tj2 = time.time() - t0
    dj = 3 * len(bL["nodes"])
    mats = [{"name": n, **{k: _mat_block(blade, n)["elastic"][k] for k in ("E", "G", "nu")},
             "rho": _mat_block(blade, n)["density"]} for n in mat_names]
    yfe = os.path.join(SCR, "solid_%s.yaml" % tag)
    write_solid_segment_yaml(yfe, res["nodes"], [list(h) for h in res["hexes"]],
                             list(hmats), np.asarray(oris), mats)
    from opensg.mesh.segment import SolidSegmentMesh
    from opensg.core.solid import compute_timo_boun
    cwd = os.getcwd(); os.chdir(SCR)
    try:
        sm = SolidSegmentMesh(yfe)
        mp, _den = sm.material_database
        t0 = time.time(); compute_timo_boun(mp, sm.left_submesh); tf = time.time() - t0
        df = 3 * sm.left_submesh["mesh"].topology.index_map(0).size_local
    finally:
        os.chdir(cwd)
    return ts, ds, tj1, tj2, dj, tf, df


def save(rows):
    np.savez(os.path.join(OUT, "timing4.npz"),
             names=[r[0] for r in rows], rows=np.array([r[1:] for r in rows], float))


NAN = float("nan")


def main():
    rows = []
    for label, shell_y, solid_y, ref in (
            ("two_cell", os.path.join(TC, "tube2cell_thin.yaml"),
             os.path.join(TC, "solid_tube2cell_thin.yaml"), None),
            ("ellipse", os.path.join(ELM, "shell_ell4cell_iso.yaml"),
             os.path.join(ELM, "solid_ell4cell_iso.yaml"), None)):
        try:
            ts, ds, _ = t_ring(shell_y, ref)
        except Exception as e:
            print("RING FAIL", label, repr(e)[:120]); ts, ds = NAN, 0
        try:
            tj1, tj2, dj, _ = t_jax_yaml(solid_y)
        except Exception as e:
            print("JAX FAIL", label, repr(e)[:120]); tj1, tj2, dj = NAN, NAN, 0
        try:
            tf, df, _ = t_fenics_2d(solid_y, label)
        except Exception as e:
            print("FENICS FAIL", label, repr(e)[:160]); tf, df = NAN, 0
        rows.append((label, ds, ts, dj, tj1, tj2, df, tf)); save(rows)
        print("%-9s: RM %6.2fs (%d dof) | JAX %6.2f/%6.2fs (%d dof) | FEniCS %6.2fs (%d dof)"
              % (label, ts, ds, tj1, tj2, dj, tf, df), flush=True)

    blade = load_blade(WINDIO)
    tot = np.zeros(3); r020 = None
    for r in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        try:
            ts, ds, tj1, tj2, dj, tf, df = iea_station(blade, r)
        except Exception as e:
            print("STATION FAIL r=%.1f %s" % (r, repr(e)[:160]), flush=True); continue
        print("  station r=%.1f: RM %5.2fs (%d) | JAX %5.2f/%5.2fs (%d) | FEniCS %5.2fs (%d)"
              % (r, ts, ds, tj1, tj2, dj, tf, df), flush=True)
        if abs(r - 0.2) < 1e-9:
            r020 = ("iea_r020", ds, ts, dj, tj1, tj2, df, tf)
            rows.append(r020); save(rows)
        tot += np.array([ts, tj2, tf])
    rows.append(("full_blade", 0, tot[0], 0, NAN, tot[1], 0, tot[2])); save(rows)
    print("full_blade: RM %6.2fs | JAX %6.2fs | FEniCS %6.2fs  (8 stations, compiled JAX)"
          % (tot[0], tot[1], tot[2]), flush=True)
    print("wrote timing4.npz")


if __name__ == "__main__":
    main()
