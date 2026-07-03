"""
taper_refine.py -- MESH-REFINEMENT probe of the m45 taper COUPLING error.

Question: the [-45] tapered tube shows the largest error in the OFF-DIAGONAL
(coupling) Timoshenko terms at strong taper (aR=0.7).  Is that

    (a) a discretization / curvature bug  -> error -> 0 as the mesh refines, or
    (b) the formulation's O(eps^2) residual -> error PLATEAUS at a finite value.

The hoop curvature k22 that feeds the general RM operator is computed by a
FINITE-DIFFERENCE vertex-turning estimate (segment_element.compute_k22) on the
NC-gon; on a coarse polygon it under-estimates 1/R, which could bias the
couplings.  Refining NC drives the discrete k22 -> -1/R exactly, so this sweep
separates the curvature-discretization error from the true taper residual.

Sweep: m45, aR=0.7, thin (t/R=0.02) + thick (t/R=0.2),
       (NC, NL) = (48,10) (96,20) (144,30) (192,40), NR=4 fixed.
Shell runs here (Windows/JAX); the matched solids run in WSL via
run_solid_study.py taper_refine.  Report = coupling %err vs NC.

    python taper_refine.py gen        write refined shell+solid meshes
    python taper_refine.py shell      run the general-RM shell on each
    python taper_refine.py report     coupling-term convergence table
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import taper_study as ts

OUT = os.path.join(HERE, "out", "taper_refine")
MESH = os.path.join(OUT, "meshes")
RES = os.path.join(OUT, "results")
for d in (MESH, RES):
    os.makedirs(d, exist_ok=True)

LEVELS = [(48, 10), (96, 20), (144, 30)]      # (NC, NL); NR=4  (192 OOMs the dense assembler)
NR = 4
AR = 0.7
REGIMES = ("thin", "thick")


def rtag(regime, nc):
    return "%s_m45_nc%03d" % (regime, nc)


def gen_level(regime, nc, nl):
    """Write shell_<tag>.yaml + solid_<tag>.yaml at this resolution (tag carries NC)."""
    tg = rtag(regime, nc)
    # taper_study.gen_case names files by tag_of(regime,mat,aR); rename to our nc-tag
    tmp = ts.gen_case(regime, "m45", AR, mesh_dir=MESH, nc=nc, nl=nl, nr=NR)
    for kind in ("shell", "solid"):
        src = os.path.join(MESH, "%s_%s.yaml" % (kind, tmp))
        dst = os.path.join(MESH, "%s_%s.yaml" % (kind, tg))
        if src != dst:
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
    return tg


def cmd_gen():
    for regime in REGIMES:
        for nc, nl in LEVELS:
            tg = gen_level(regime, nc, nl)
            print("mesh", tg, "(NC=%d NL=%d NR=%d)" % (nc, nl, NR))


def cmd_shell():
    for regime in REGIMES:
        for nc, nl in LEVELS:
            tg = rtag(regime, nc)
            rL, S6, rR = ts.shell_solve(tg, shear="mitc4_both", mesh_dir=MESH, res_dir=RES)
            print("shell %-16s taper diag(x1e9) %s" % (tg, np.array2string(np.diag(S6) / 1e9, precision=3)))


# off-diagonal (coupling) terms that are non-trivially excited by the [-45] taper
COUP = [(0, 3), (1, 5), (2, 4), (0, 2), (0, 4), (3, 4)]   # C14,C26,C35,C13,C15,C45 (1-indexed)
DIAG = [(i, i) for i in range(6)]
ENG = {0: "EA", 1: "GA2", 2: "GA3", 3: "GJ", 4: "EI2", 5: "EI3"}


def _load(tg, part):
    Sh = np.load(os.path.join(RES, "rm_%s_%s.npy" % (tg, part)))
    So = np.load(os.path.join(RES, "solid_%s_%s.npy" % (tg, part)))
    return 0.5 * (Sh + Sh.T), 0.5 * (So + So.T)


def cmd_report():
    for regime in REGIMES:
        print("\n===== m45 %s  aR=0.7  TAPERED SEGMENT : term %%err vs mesh (NC) =====" % regime.upper())
        # header
        cols = DIAG + COUP
        names = ["C%d%d" % (i + 1, j + 1) for i, j in cols]
        print("%-6s " % "NC" + " ".join("%8s" % n for n in names))
        base = None
        for nc, nl in LEVELS:
            tg = rtag(regime, nc)
            try:
                Sh, So = _load(tg, "seg")
            except FileNotFoundError:
                print("%-6d  (solid missing)" % nc); continue
            errs = []
            for i, j in cols:
                errs.append(100 * (Sh[i, j] - So[i, j]) / So[i, j] if abs(So[i, j]) > 1e-6 * abs(So[0, 0]) else float("nan"))
            print("%-6d " % nc + " ".join("%+7.1f%%" % e for e in errs))
        # richardson-ish read: does the coupling error shrink like 1/NC^2 (bug) or flatten (formulation)?
        print("  [diag C11..C66 are EA GA2 GA3 GJ EI2 EI3 ; couplings C14 C26 C35 C13 C15 C45]")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gen"
    {"gen": cmd_gen, "shell": cmd_shell, "report": cmd_report}[cmd]()
