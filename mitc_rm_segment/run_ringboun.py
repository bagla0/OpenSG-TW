"""run_ringboun.py -- WHY the boundary RING needs the gamma_23 tie (and the segment
does not): square vs ellipse boundary, full integration vs gamma_23-tied, t sweep.

The ring is the span-invariant SG: the top node row is DOF-mapped onto the bottom
row, so every fluctuation x1-derivative vanishes identically and gamma_13 is
ALGEBRAIC in the rotations (no locking pairing possible).  gamma_23 keeps the hoop
flux y_i w_{i|2} paired with the linear rotation trace -x_{i;1}omega_i: on a CURVED
wall the section response contains genuine hoop wall bending (breathing/ovalization
under extension/bending Poisson effects), so the thin limit enforces the discrete
Kirchhoff constraint w_{n,s}=theta_s on the self-contained ring fluctuation -> the
classical linear Timoshenko-element lock.  On FLAT walls (square) the fluctuation
is membrane-dominated (hoop bending only in corner layers) and full integration
survives.  Detector: error vs the self-converged tied ring at nc=384, swept over
t = 0.02 / 0.002 / 0.0002 and nc = 48 / 96 / 192 -- locking = growth with 1/t at
fixed nc, cured by refinement; discretization error = t-independent.

    python run_ringboun.py
"""
import io, os, sys, json, math, contextlib
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
BENCH = os.path.join(REPO, "examples", "data", "benchmark")
OUT = os.path.join(HERE, "out", "ringboun")
os.makedirs(OUT, exist_ok=True)

import taper_study as ts
import taper_square as tsq
from boundary_from_yaml import extract
from segment_element import compute_k22
from solve_segment_jax import _material_by_section
from run_ring_indep import ring_indep

L, A0, A1, B0, B1 = 2.0, 1.0, 0.65, 0.6, 0.42
ISO = dict(name="iso", rho=1800.0, E=[70e9] * 3, G=[70e9 / 2.6] * 3, nu=[0.3] * 3)
ANI = dict(name="ani", rho=1800.0, E=[37e9, 9e9, 9e9], G=[4e9] * 3, nu=[0.3] * 3)
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def gen_ellipse(mesh_dir, nc, nl, T, mat):
    """Tapered ellipse shell yaml (same geometry as run_ellipse; parametric T, mat)."""
    os.makedirs(mesh_dir, exist_ok=True)
    M = ISO if mat == "iso" else ANI
    ang = 0.0 if mat == "iso" else -45.0
    Z = [L * i / nl for i in range(nl + 1)]

    def ab(z):
        return A0 + (A1 - A0) * z / L, B0 + (B1 - B0) * z / L

    def frame(th, z):
        a, b = ab(z)
        da, db = (A1 - A0) / L, (B1 - B0) / L
        gen = np.array([da * math.cos(th), db * math.sin(th), 1.0])
        hoop = np.array([-a * math.sin(th), b * math.cos(th), 0.0])
        a2 = hoop / np.linalg.norm(hoop)
        a1 = gen - (gen @ a2) * a2; a1 /= np.linalg.norm(a1)
        e3 = np.cross(a1, a2); e3 /= np.linalg.norm(e3)
        n2d = np.array([b * math.cos(th), a * math.sin(th), 0.0]); n2d /= np.linalg.norm(n2d)
        if e3 @ (-n2d) < 0:
            e3 = -e3; a1 = -a1
        return a1, a2, e3

    snodes, squads, soris = [], [], []
    for z in Z:
        a, b = ab(z)
        for k in range(nc):
            th = 2 * math.pi * k / nc
            snodes.append([a * math.cos(th), b * math.sin(th), float(z)])
    for i in range(nl):
        for k in range(nc):
            k1 = (k + 1) % nc
            squads.append([i * nc + k + 1, i * nc + k1 + 1,
                           (i + 1) * nc + k1 + 1, (i + 1) * nc + k + 1])
            a1, a2, e3 = frame(2 * math.pi * (k + 0.5) / nc, (Z[i] + Z[i + 1]) / 2)
            soris.append(a1.tolist() + a2.tolist() + e3.tolist())
    shell = {"nodes": snodes, "elements": squads,
             "sections": [{"elementSet": "wall", "layup": [[M["name"], T, ang]]}],
             "sets": {"element": [{"name": "wall", "labels": list(range(1, len(squads) + 1))}]},
             "materials": [{"name": M["name"], "density": M["rho"],
                            "elastic": {"E": M["E"], "G": M["G"], "nu": M["nu"]}}],
             "elementOrientations": soris}
    fn = os.path.join(mesh_dir, "shell_ell.yaml")
    yaml.safe_dump(shell, open(fn, "w"), default_flow_style=None, sort_keys=False)
    return fn


def ring_from_yaml(shell_yaml, tagnpz, shear):
    with contextlib.redirect_stdout(io.StringIO()):
        extract(shell_yaml, tagnpz)
    b = np.load(tagnpz, allow_pickle=True)
    ax = int(b["axis"]); cross = [j for j in range(3) if j != ax]
    rx = np.asarray(b["L_x"]); rc = np.asarray(b["L_cells"])
    rs = np.asarray(b["L_subdom"]); re3 = np.asarray(b["L_e3"])
    D_by, G_by = _material_by_section(json.loads(str(b["sections"])),
                                      json.loads(str(b["materials"])), center_ref=True)
    kr = compute_k22(rx[rc].mean(1), np.asarray(b["L_e2"]), re3, rc)
    C = ring_indep(rx, rc, rs, re3, D_by, G_by, kr, ax, cross, shear=shear)
    return 0.5 * (C + C.T)


def mesh_for(geom, mat, tR, nc):
    mdir = os.path.join(OUT, "%s_%s_t%s_nc%d" % (geom, mat, str(tR).replace(".", "p"), nc))
    if geom == "square":
        old = ts.THICK["thin"]; ts.THICK["thin"] = tR
        tsq.gen_square_case("thin", mat, 0.7, mesh_dir=mdir, nc=nc, nl=2)
        ts.THICK["thin"] = old
        return os.path.join(mdir, "shell_thin_%s_aR070.yaml" % mat)
    return gen_ellipse(mdir, nc, 2, tR, mat)      # ellipse: t absolute (b0=0.6)


def main():
    print("BOUNDARY RING shear-scheme study: full integration vs gamma23-tied")
    print("errors vs SELF-CONVERGED tied ring at nc=384 (same t); solid L at t=0.02\n")
    for geom in ("square", "ellipse"):
        for mat in ("iso", "m45"):
            for tR in (0.02, 0.002, 0.0002):
                yref = mesh_for(geom, mat, tR, 384)
                npz = os.path.join(OUT, "b_%s_%s_ref.npz" % (geom, mat))
                Cref = ring_from_yaml(yref, npz, "mitc4_g23")
                Cchk = ring_from_yaml(yref, npz, "full")
                dev = max(abs(100 * (Cchk[i, i] - Cref[i, i]) / Cref[i, i]) for i in range(6))
                print("== %s %s t=%g ==  (full@384 vs tied@384 max diag dev %.2f%%)"
                      % (geom, mat, tR, dev))
                print("%4s %-10s" % ("nc", "scheme") + "".join("%8s" % l for l in LBL))
                for nc in (48, 96, 192):
                    ym = mesh_for(geom, mat, tR, nc)
                    for sch, nm in (("full", "full"), ("mitc4_g23", "g23-tied")):
                        C = ring_from_yaml(ym, npz, sch)
                        e = [100 * (C[i, i] - Cref[i, i]) / Cref[i, i] for i in range(6)]
                        print("%4d %-10s" % (nc, nm) + "".join("%+7.2f " % v for v in e))
                        sys.stdout.flush()
                if tR == 0.02:                      # solid-reference cross-check
                    So = None
                    if geom == "square":
                        f = os.path.join(BENCH, "taper_square_solid_%s.npz" % mat)
                        if os.path.exists(f):
                            s = np.load(f, allow_pickle=True)
                            So = 0.5 * (s["thin_%s_aR070_L" % mat] + s["thin_%s_aR070_L" % mat].T)
                    elif mat == "m45":
                        f = os.path.join(HERE, "out", "ellipse", "solid_ref.npz")
                        if os.path.exists(f):
                            s = np.load(f); So = 0.5 * (s["L"] + s["L"].T)
                    if So is not None:
                        e = [100 * (Cref[i, i] - So[i, i]) / So[i, i] for i in range(6)]
                        print("     tied@384 vs 3-D solid L:      "
                              + "".join("%+7.2f " % v for v in e))
                print()


if __name__ == "__main__":
    main()
