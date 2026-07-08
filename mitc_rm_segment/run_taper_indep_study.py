"""run_taper_indep_study.py -- ALL-6-DOF tapered shell homogenization study.

Single model: the constrained (independent-omega3) six-parameter RM element for BOTH
the boundary rings and the tapered segment (element-constant Lagrange multipliers,
full shear integration), compared against the FEniCS 3-D solid.

Runs all 8 cases {square, circle} x {thin, thick} x {iso, m45} at strong taper aR=0.7.
Writes taper_indep_study/taper_indep_results.dat (rings + segment full 6x6 + %err,
per-stage timing table) and bundles the mesh files + this script.

    python run_taper_indep_study.py
"""
import os, sys, re, shutil, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
import taper_study as ts
import run_indep as ri

BENCH = os.path.join(REPO, "examples", "data", "benchmark")
OUT = os.path.join(HERE, "taper_indep_study")
MESHOUT = os.path.join(OUT, "meshes")
for d in (OUT, MESHOUT):
    os.makedirs(d, exist_ok=True)

GEOM = {
    "square": dict(mesh=os.path.join(HERE, "out", "taper_square", "meshes"),
                   res=os.path.join(HERE, "out", "taper_square", "results"),
                   npz="taper_square_solid_%s.npz", timings="taper_square_solid_timings.txt"),
    "circle": dict(mesh=os.path.join(HERE, "out", "taper_study", "meshes"),
                   res=os.path.join(HERE, "out", "taper_study", "results"),
                   npz="taper_study_solid_%s.npz", timings="taper_study_solid_timings.txt"),
}
LBL = ["C11 EA", "C22 GA2", "C33 GA3", "C44 GJ", "C55 EI2", "C66 EI3"]
AR = 0.7


def solid_times(fname):
    """tag -> (boun_s, taper_s), fastest recorded run per tag."""
    out = {}
    for line in open(os.path.join(BENCH, fname)):
        m = re.match(r"solid (\S+)\s+boun (\d+)s taper (\d+)s", line)
        if m:
            tg, tb, tt = m.group(1), int(m.group(2)), int(m.group(3))
            if tg not in out or tb + tt < sum(out[tg]):
                out[tg] = (tb, tt)
    return out


def run():
    cases = []
    for geom in ("square", "circle"):
        G = GEOM[geom]
        stimes = solid_times(G["timings"])
        for regime in ("thin", "thick"):
            for mat in ("iso", "m45"):
                tg = ts.tag_of(regime, mat, AR)
                b = np.load(os.path.join(BENCH, G["npz"] % mat), allow_pickle=True)
                r = ri.shell_solve_lagrange(tg, G["mesh"], G["res"], return_full=True)
                for kind in ("shell", "solid"):
                    src = os.path.join(G["mesh"], "%s_%s.yaml" % (kind, tg))
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(MESHOUT, "%s_%s_%s.yaml" % (geom, kind, tg)))
                cases.append(dict(geom=geom, regime=regime, mat=mat, tg=tg, r=r,
                                  SoL=0.5 * (b[tg + "_L"] + b[tg + "_L"].T),
                                  SoR=0.5 * (b[tg + "_R"] + b[tg + "_R"].T),
                                  So=0.5 * (b[tg + "_seg"] + b[tg + "_seg"].T),
                                  st=stimes.get(tg, (float("nan"), float("nan")))))
                print("done", geom, tg, "(rings %.1fs seg %.1fs)" % (r["t_rings"], r["t_seg"]))
                sys.stdout.flush()

    with open(os.path.join(OUT, "taper_indep_results.dat"), "w") as f:
        def w(s=""):
            f.write(s + "\n")
        w("=" * 88)
        w("TAPERED SHELL HOMOGENIZATION -- all-6-DOF constrained Reissner-Mindlin model")
        w("=" * 88)
        w("Model  : independent-drilling (6-DOF/node) RM shell, in-plane symmetry enforced by")
        w("         element-constant Lagrange multipliers, FULL shear integration; the SAME")
        w("         element solves the boundary rings (wrapped strip) and the tapered segment;")
        w("         ring warping fields (incl. omega_3) are the segment's Dirichlet data.")
        w("Cases  : {square, circle} x {thin t/R=0.02, thick t/R=0.2} x {iso, m45 [-45] ply}")
        w("Taper  : STRONG, aR=0.7 (radius 1.0 -> 0.7 over L=2.0, mid-surface reference)")
        w("Solver : run_indep.shell_solve_lagrange + run_ring_indep.ring_indep (segment_indep ops)")
        w("Meshes : ./meshes/{square,circle}_{shell,solid}_<tag>.yaml (48 hoop x 10 axial shell;")
        w("         48 x 10 x 4 solid).  Reference: FEniCS 3-D solid segment homogenization.")
        w("")
        w("=" * 88)
        w("SUMMARY : tapered-segment diagonal %err vs 3-D solid")
        w("=" * 88)
        w("%-22s %8s %8s %8s %8s %8s %8s"
          % ("case", "EA", "GA2", "GA3", "GJ", "EI2", "EI3"))
        for c in cases:
            S, So = c["r"]["S6"], c["So"]
            e = [100 * (S[i, i] - So[i, i]) / So[i, i] for i in range(6)]
            w("%-22s %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%%"
              % ("%s_%s_%s" % (c["geom"], c["regime"], c["mat"]), *e))
        w("")
        w("=" * 88)
        w("TIMING : wall-clock seconds per case (single core)")
        w("=" * 88)
        w("%-22s %9s %9s %9s %9s | %9s %9s %9s"
          % ("case", "extract", "rings", "segment", "shell", "sol.boun", "sol.taper", "solid"))
        for c in cases:
            r = c["r"]; tb, tt = c["st"]
            tot = r["t_extract"] + r["t_rings"] + r["t_seg"]
            w("%-22s %9.1f %9.1f %9.1f %9.1f | %9.0f %9.0f %9.0f"
              % ("%s_%s_%s" % (c["geom"], c["regime"], c["mat"]),
                 r["t_extract"], r["t_rings"], r["t_seg"], tot, tb, tt, tb + tt))
        w("")
        for c in cases:
            r = c["r"]
            w("-" * 88)
            w("CASE  %s_%s" % (c["geom"], c["tg"]))
            w("-" * 88)
            for part, S, So in (("L ring ", r["C6L"], c["SoL"]),
                                ("SEGMENT", r["S6"], c["So"]),
                                ("R ring ", r["C6R"], c["SoR"])):
                w("  == %s ==" % part)
                w("  %-9s %14s %14s %9s" % ("term", "solid", "shell", "%err"))
                for i in range(6):
                    e = 100 * (S[i, i] - So[i, i]) / So[i, i] if So[i, i] else float("nan")
                    w("  %-9s %14.5e %14.5e %+8.1f%%" % (LBL[i], So[i, i], S[i, i], e))
                thr = 1e-3 * abs(np.diag(So)).max()
                for i in range(6):
                    for j in range(i + 1, 6):
                        if abs(So[i, j]) > thr:
                            e = 100 * (S[i, j] - So[i, j]) / So[i, j]
                            w("  C%d%d       %14.5e %14.5e %+8.1f%%" % (i + 1, j + 1, So[i, j], S[i, j], e))
    shutil.copy(os.path.abspath(__file__), os.path.join(OUT, "run_taper_indep_study.py"))
    print("\nwrote", os.path.join(OUT, "taper_indep_results.dat"))


if __name__ == "__main__":
    run()
