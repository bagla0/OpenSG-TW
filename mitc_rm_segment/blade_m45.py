"""blade_m45.py -- uniform-m45 SHELL homogenization of BAR-URC blade segments 5/12/15.

Overrides every layup of the real segment with a SINGLE [-45] ply (constant t, ANI
material) so the SG is uniform along the span => the frozen right-boundary layup ==
the true layup (one-to-one with a solid).  Reports, per segment, the general-RM
Timoshenko 6x6 at the LEFT ring, RIGHT ring, their AVERAGE, and the TAPERED segment,
plus %diff(tapered vs avg-of-ends).

    python blade_m45.py            # seg 5,12,15  t=0.04
    python blade_m45.py 0.03 5 12  # custom t and segment list
"""
import os, sys, json, time
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
import taper_study as ts

D3 = os.path.join(REPO, "examples", "data", "3d_yaml")
TMPM = os.path.join(HERE, "out", "blade_m45", "meshes")
TMPR = os.path.join(HERE, "out", "blade_m45", "results")
for d in (TMPM, TMPR):
    os.makedirs(d, exist_ok=True)

ANI = dict(name="ani", rho=1800.0, E=[37e9, 9e9, 9e9], G=[4e9, 4e9, 4e9], nu=[0.3, 0.3, 0.3])
LBL = ["EA (C11)", "GA2 (C22)", "GA3 (C33)", "GJ (C44)", "EI2 (C55)", "EI3 (C66)"]


def make_m45_shell(seg_id, t, out_yaml):
    """Load the real BAR-URC shell segment, replace ALL layups with one [-45] ANI
    ply of thickness t, keep the geometry + per-element orientations, write shell yaml."""
    src = os.path.join(D3, "BAR_URC_numEl_52_segment_%d.yaml" % seg_id)
    d = yaml.safe_load(open(src))
    ne = len(d["elements"])
    shell = {
        "nodes": d["nodes"],
        "elements": d["elements"],
        "sections": [{"elementSet": "all", "layup": [["ani", float(t), -45.0]]}],
        "sets": {"element": [{"name": "all", "labels": list(range(1, ne + 1))}]},
        "materials": [{"name": ANI["name"], "density": ANI["rho"],
                       "elastic": {"E": ANI["E"], "G": ANI["G"], "nu": ANI["nu"]}}],
        "elementOrientations": d["elementOrientations"],
    }
    yaml.safe_dump(shell, open(out_yaml, "w"), default_flow_style=None, sort_keys=False)
    return ne


def run(seg_ids=(5, 12, 15), t=0.04):
    print("### BAR-URC blade segments, UNIFORM m45 ([-45] ANI, t=%.3f) -- SHELL general-RM ###" % t)
    for seg in seg_ids:
        tag = "barseg%d" % seg
        yfn = os.path.join(TMPM, "shell_%s.yaml" % tag)
        ne = make_m45_shell(seg, t, yfn)
        t0 = time.time()
        C6L, S6, C6R = ts.shell_solve(tag, shear="mitc4_both", mesh_dir=TMPM, res_dir=TMPR)
        dt = time.time() - t0
        C6L = 0.5 * (C6L + C6L.T); C6R = 0.5 * (C6R + C6R.T); S6 = 0.5 * (S6 + S6.T)
        avg = 0.5 * (C6L + C6R)
        print("\n===== segment %d  (%d elems, %.1fs) =====" % (seg, ne, dt))
        print("%-11s %12s %12s %12s %12s | %9s" % ("term", "L ring", "R ring", "avg(L,R)", "TAPERED", "tap-vs-avg"))
        for i in range(6):
            d = 100 * (S6[i, i] - avg[i, i]) / avg[i, i] if avg[i, i] else float("nan")
            print("%-11s %12.4e %12.4e %12.4e %12.4e | %+8.1f%%"
                  % (LBL[i], C6L[i, i] / 1e9, C6R[i, i] / 1e9, avg[i, i] / 1e9, S6[i, i] / 1e9, d))
        # save for the tutorial / solid comparison
        np.savez(os.path.join(TMPR, "shell_%s_m45.npz" % tag), L=C6L, R=C6R, avg=avg, seg=S6, t=t, ne=ne)


if __name__ == "__main__":
    args = sys.argv[1:]
    t = float(args[0]) if args else 0.04
    segs = tuple(int(x) for x in args[1:]) if len(args) > 1 else (5, 12, 15)
    run(segs, t)
