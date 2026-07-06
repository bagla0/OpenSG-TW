"""run_taper_indep_study.py -- taper shell homogenization : independent-omega3 (Lagrange) GA3 fix.

Runs all 8 cases  {square, circle} x {thin, thick} x {iso, m45}  at STRONG taper aR=0.7 :
  general-RM (omega3 eliminated) vs independent-omega3 RM (Lagrange constraint) vs 3-D solid.
Writes taper_indep_study/taper_indep_results.dat, and bundles the mesh files + this script.

    python run_taper_indep_study.py
"""
import os, sys, shutil, time
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
                   npz="taper_square_solid_%s.npz"),
    "circle": dict(mesh=os.path.join(HERE, "out", "taper_study", "meshes"),
                   res=os.path.join(HERE, "out", "taper_study", "results"),
                   npz="taper_study_solid_%s.npz"),
}
LBL = ["C11 EA", "C22 GA2", "C33 GA3", "C44 GJ", "C55 EI2", "C66 EI3"]
AR = 0.7


def run():
    cases = []
    for geom in ("square", "circle"):
        G = GEOM[geom]
        for regime in ("thin", "thick"):
            for mat in ("iso", "m45"):
                tg = ts.tag_of(regime, mat, AR)
                b = np.load(os.path.join(BENCH, G["npz"] % mat), allow_pickle=True)
                So = 0.5 * (b[tg + "_seg"] + b[tg + "_seg"].T)
                t0 = time.time()
                _, Sg, _ = ts.shell_solve(tg, shear="mitc4_both", mesh_dir=G["mesh"], res_dir=G["res"])
                tgen = time.time() - t0
                Sg = 0.5 * (Sg + Sg.T)
                t0 = time.time()
                Si = ri.shell_solve_lagrange(tg, G["mesh"], G["res"])
                tind = time.time() - t0
                for kind in ("shell", "solid"):
                    src = os.path.join(G["mesh"], "%s_%s.yaml" % (kind, tg))
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(MESHOUT, "%s_%s_%s.yaml" % (geom, kind, tg)))
                cases.append(dict(geom=geom, regime=regime, mat=mat, tg=tg,
                                  So=So, Sg=Sg, Si=Si, tgen=tgen, tind=tind))
                print("done", geom, tg, "(gen %.0fs, indep %.0fs)" % (tgen, tind)); sys.stdout.flush()

    with open(os.path.join(OUT, "taper_indep_results.dat"), "w") as f:
        def w(s=""):
            f.write(s + "\n")
        w("=" * 88)
        w("TAPER SHELL HOMOGENIZATION -- independent-omega3 (Lagrange) transverse-shear (GA3) fix")
        w("=" * 88)
        w("Cases   : {square, circle} x {thin t/R=0.02, thick t/R=0.2} x {iso, m45 single-ply [-45]}")
        w("Taper   : STRONG, aR=0.7  (radius 1.0 -> 0.7 linearly over L=2.0, center-reference mesh)")
        w("Methods :")
        w("  solid     = FEniCS 3-D solid tapered segment (reference)")
        w("  general   = RM shell, omega_3 ELIMINATED (segment_element_general.py) -- the GA3-deficient op")
        w("  indep-lag = RM shell, omega_3 as INDEPENDENT 6th DOF; in-plane symmetry DR=0 imposed EXACTLY")
        w("              by nodal LAGRANGE MULTIPLIERS (no penalty, no tuning) -- the fix")
        w("Run script : run_taper_indep_study.py            (bundled in this folder)")
        w("RM code    : segment_indep.py            (6-DOF operator quad_ops_indep + assemble_constraint)")
        w("             run_indep.py::shell_solve_lagrange  (augmented-KKT solve + MSG V0/V1 condensation)")
        w("Mesh gen   : taper_square.py (square), taper_study.py (circle)")
        w("Meshes     : ./meshes/{square,circle}_{shell,solid}_<tag>.yaml  (bundled)")
        w("Env        : C:\\conda_envs\\opensg_2_0_env")
        w("")
        w("=" * 88)
        w("SUMMARY : transverse-shear GA3 (C33) and GA2 (C22) %err vs 3-D solid")
        w("=" * 88)
        w("%-22s %11s %11s   %11s %11s" % ("case", "GA3 general", "GA3 indep", "GA2 general", "GA2 indep"))
        for c in cases:
            So, Sg, Si = c["So"], c["Sg"], c["Si"]
            def e(M, i):
                return 100 * (M[i, i] - So[i, i]) / So[i, i] if So[i, i] else float("nan")
            w("%-22s %+10.1f%% %+10.1f%%   %+10.1f%% %+10.1f%%"
              % ("%s_%s_%s" % (c["geom"], c["regime"], c["mat"]), e(Sg, 2), e(Si, 2), e(Sg, 1), e(Si, 1)))
        w("")
        w("(square thin is the pathological flat-wall case: general GA3 -24%/-40%; indep ~ -3..-4%.")
        w(" circle is curved -> already good under general; indep leaves it unchanged = no regression.)")
        w("")
        for c in cases:
            So, Sg, Si = c["So"], c["Sg"], c["Si"]
            w("-" * 88)
            w("CASE  %s_%s   (general %.0fs, indep-lagrange %.0fs)" % (c["geom"], c["tg"], c["tgen"], c["tind"]))
            w("-" * 88)
            w("%-9s %14s %14s %14s | %9s %9s"
              % ("term", "solid", "general", "indep-lag", "gen %err", "ind %err"))
            for i in range(6):
                eg = 100 * (Sg[i, i] - So[i, i]) / So[i, i] if So[i, i] else float("nan")
                ei = 100 * (Si[i, i] - So[i, i]) / So[i, i] if So[i, i] else float("nan")
                w("%-9s %14.5e %14.5e %14.5e | %+8.1f%% %+8.1f%%" % (LBL[i], So[i, i], Sg[i, i], Si[i, i], eg, ei))
            thr = 1e-3 * abs(np.diag(So)).max()
            hdr = False
            for i in range(6):
                for j in range(i + 1, 6):
                    if abs(So[i, j]) > thr:
                        if not hdr:
                            w("  off-diagonal couplings (|C_ij| > 0.1%% of max diag):"); hdr = True
                        eg = 100 * (Sg[i, j] - So[i, j]) / So[i, j]
                        ei = 100 * (Si[i, j] - So[i, j]) / So[i, j]
                        w("    C%d%d  solid=%+12.5e  general %+7.1f%%   indep %+7.1f%%" % (i + 1, j + 1, So[i, j], eg, ei))
    shutil.copy(os.path.abspath(__file__), os.path.join(OUT, "run_taper_indep_study.py"))
    print("\nwrote", os.path.join(OUT, "taper_indep_results.dat"))
    print(open(os.path.join(OUT, "taper_indep_results.dat")).read())


if __name__ == "__main__":
    run()
