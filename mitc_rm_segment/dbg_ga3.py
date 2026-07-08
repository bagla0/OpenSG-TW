"""dbg_ga3.py -- reproduce & localize the thin-square GA3 (C33) deficit.

Usage:  python dbg_ga3.py peek        # show solid-ref npz keys/shapes
        python dbg_ga3.py base        # reproduce baseline deficit (thin iso+m45, aR070)
"""
import os, sys, json, math
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)

BENCH = os.path.join(REPO, "examples", "data", "benchmark")


def peek():
    for mat in ("iso", "m45"):
        f = os.path.join(BENCH, "taper_square_solid_%s.npz" % mat)
        b = np.load(f, allow_pickle=True)
        print("=== %s ===" % mat)
        for k in b.files:
            v = b[k]
            try:
                print("  %-24s %s %s" % (k, getattr(v, "shape", ""), getattr(v, "dtype", "")))
            except Exception:
                print("  %-24s (obj)" % k)


def _fmt6(M, scale=1e9):
    return "\n".join("  " + "  ".join("%+11.4e" % (M[i, j] / scale) for j in range(6)) for i in range(6))


LBL = ["C11", "C22", "C33", "C44", "C55", "C66"]


def _cmp(name, So, Sh):
    So = 0.5 * (So + So.T); Sh = 0.5 * (Sh + Sh.T)
    print("  -- %s : diag  solid | RM | %%err --" % name)
    for i in range(6):
        e = 100 * (Sh[i, i] - So[i, i]) / So[i, i] if So[i, i] else float("nan")
        flag = "  <<<" if abs(e) > 10 else ""
        print("     %-4s  %+11.4e  %+11.4e  %+7.1f%%%s" % (LBL[i], So[i, i] / 1e9, Sh[i, i] / 1e9, e, flag))
    # C36 (=GA3-EI2 coupling) is the reported tracker
    e36 = 100 * (Sh[2, 5] - So[2, 5]) / So[2, 5] if So[2, 5] else float("nan")
    print("     C36   %+11.4e  %+11.4e  %+7.1f%%   (GA3-EI2 coupling)" % (So[2, 5] / 1e9, Sh[2, 5] / 1e9, e36))


def base(regimes=("thin",), aRs=(0.7,), shear="mitc4_both", mesh_dir=None, nl=None, tag=""):
    import taper_study as ts
    import taper_square as tsq
    MESH = mesh_dir or os.path.join(HERE, "out", "taper_square", "meshes")
    RES = os.path.join(HERE, "out", "taper_square", "results")
    os.makedirs(MESH, exist_ok=True)
    for mat in ("iso", "m45"):
        f = os.path.join(BENCH, "taper_square_solid_%s.npz" % mat)
        b = np.load(f, allow_pickle=True)
        for regime in regimes:
            for aR in aRs:
                tg = ts.tag_of(regime, mat, aR)
                # (re)generate mesh if a custom nl or mesh_dir is requested
                if nl is not None or mesh_dir is not None:
                    tsq.gen_square_case(regime, mat, aR, mesh_dir=MESH, nl=nl)
                print("\n########## %s  shear=%s  nl=%s %s ##########" % (tg, shear, nl, tag))
                rL, S6, rR = ts.shell_solve(tg, shear=shear, mesh_dir=MESH, res_dir=RES)
                _cmp("SEG   ", b["%s_seg" % tg], S6)
                _cmp("ringL ", b["%s_L" % tg], rL)
                _cmp("ringR ", b["%s_R" % tg], rR)


def refine(mat="iso", regime="thin", aR=0.7, nls=(10, 20, 40, 80, 160), shear="mitc4_both"):
    """Axial (eta) refinement sweep: does SEG C33 recover as the trapezoid -> parallelogram?
    NL up isolates quad DISTORTION (shrinks) from generator TILT (unchanged)."""
    import taper_study as ts
    import taper_square as tsq
    SCR = os.path.join(HERE, "out", "taper_square", "refine_meshes")
    RES = os.path.join(HERE, "out", "taper_square", "results")
    os.makedirs(SCR, exist_ok=True)
    f = os.path.join(BENCH, "taper_square_solid_%s.npz" % mat)
    b = np.load(f, allow_pickle=True)
    tg = ts.tag_of(regime, mat, aR)
    So = 0.5 * (b["%s_seg" % tg] + b["%s_seg" % tg].T)
    print("### refine %s shear=%s : SEG C33 vs solid=%.5e (x1e9) ###" % (tg, shear, So[2, 2] / 1e9))
    print("%-6s %12s %12s %8s %8s %8s %8s" % ("NL", "C33_RM", "C33_err%", "C22_err%", "C36_err%", "C44_err%", "C55_err%"))
    for nl in nls:
        tsq.gen_square_case(regime, mat, aR, mesh_dir=SCR, nl=nl)
        rL, S6, rR = ts.shell_solve(tg, shear=shear, mesh_dir=SCR, res_dir=RES)
        S6 = 0.5 * (S6 + S6.T)
        def er(i, j=None):
            j = i if j is None else j
            return 100 * (S6[i, j] - So[i, j]) / So[i, j] if So[i, j] else float("nan")
        print("%-6d %12.5e %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%% %+7.1f%%"
              % (nl, S6[2, 2] / 1e9, er(2), er(1), er(2, 5), er(3), er(4)))


def arsweep(mat="iso", regime="thin", shear="mitc4_both"):
    """Taper-rate sweep aR=1.0..0.7 : how does SEG C33 deficit grow with taper?
    aR=1 is prismatic (formulation floor); the growth ORDER in (1-aR) reveals
    whether the missing term is O(dRdz) or O(dRdz^2)."""
    import taper_study as ts
    MESH = os.path.join(HERE, "out", "taper_square", "meshes")
    RES = os.path.join(HERE, "out", "taper_square", "results")
    import taper_square as tsq
    f = os.path.join(BENCH, "taper_square_solid_%s.npz" % mat)
    b = np.load(f, allow_pickle=True)
    print("### aR sweep %s_%s shear=%s : SEG C33 & C22 err vs solid ###" % (regime, mat, shear))
    print("%-6s %8s %10s %10s %10s %10s" % ("aR", "dRdz", "C33_err%", "C22_err%", "C36_err%", "C44_err%"))
    for aR in (1.0, 0.95, 0.9, 0.8, 0.7):
        tg = ts.tag_of(regime, mat, aR)
        if not os.path.exists(os.path.join(MESH, "shell_%s.yaml" % tg)):
            tsq.gen_square_case(regime, mat, aR, mesh_dir=MESH)
        rL, S6, rR = ts.shell_solve(tg, shear=shear, mesh_dir=MESH, res_dir=RES)
        S6 = 0.5 * (S6 + S6.T); So = 0.5 * (b["%s_seg" % tg] + b["%s_seg" % tg].T)
        def er(i, j=None):
            j = i if j is None else j
            return 100 * (S6[i, j] - So[i, j]) / So[i, j] if So[i, j] else float("nan")
        print("%-6.2f %8.4f %+9.1f%% %+9.1f%% %+9.1f%% %+9.1f%%"
              % (aR, (aR - 1.0) / 2.0, er(2), er(1), er(2, 5), er(3)))


def epssweep(regime="thin", aR=0.7):
    """C33_EPS (Tikhonov drilling-drop scale) sweep on the thin square, iso+m45.
    If the GA3 deficit is the drilling-elimination drop on the C33~0 GA3-carrying
    walls, changing eps moves C33.  Watch C44 (GJ) for the folded-wall blow-up."""
    import importlib
    import taper_study as ts
    import segment_element_general as seg
    MESH = os.path.join(HERE, "out", "taper_square", "meshes")
    RES = os.path.join(HERE, "out", "taper_square", "results")
    for mat in ("iso", "m45"):
        f = os.path.join(BENCH, "taper_square_solid_%s.npz" % mat)
        b = np.load(f, allow_pickle=True)
        tg = ts.tag_of(regime, mat, aR)
        So = 0.5 * (b["%s_seg" % tg] + b["%s_seg" % tg].T)
        print("\n### C33_EPS sweep %s : C33 solid=%.4e C44(GJ) solid=%.4e (x1e9) ###"
              % (tg, So[2, 2] / 1e9, So[3, 3] / 1e9))
        print("%-8s %10s %10s %10s %12s" % ("C33_EPS", "C33_err%", "C22_err%", "C44_err%", "C44_RM(x1e9)"))
        for eps in (0.3, 0.1, 0.05, 0.02, 0.005):
            seg.C33_EPS = eps
            rL, S6, rR = ts.shell_solve(tg, shear="mitc4_both", mesh_dir=MESH, res_dir=RES)
            S6 = 0.5 * (S6 + S6.T)
            def er(i):
                return 100 * (S6[i, i] - So[i, i]) / So[i, i] if So[i, i] else float("nan")
            print("%-8.3f %+9.1f%% %+9.1f%% %+9.1f%% %12.4e" % (eps, er(2), er(1), er(3), S6[3, 3] / 1e9))
        seg.C33_EPS = 0.1


def _set(seg, bge=1, bgl=1, y1=1, lam=1, kg=1, gsc=1):
    seg.SH_ABL_BGE, seg.SH_ABL_BGL, seg.SH_ABL_Y1 = float(bge), float(bgl), float(y1)
    seg.LAMBDA_ON, seg.KG_ABL, seg.G_SHEAR_SCALE = float(lam), float(kg), float(gsc)


def ablate(regime="thin", aR=0.7):
    """Comprehensive operator ablation: which taper-activated block robs GA3?
    Shear ablation already ~null -> now test curvature-drilling (LAMBDA), geodesic
    curvature (kg), and whether C33 is even shell-shear-G-driven (G scale)."""
    import taper_study as ts
    import segment_element_general as seg
    MESH = os.path.join(HERE, "out", "taper_square", "meshes")
    RES = os.path.join(HERE, "out", "taper_square", "results")
    #     name                bge bgl y1 lam kg  gsc
    configs = [("baseline",     1,  1, 1,  1,  1,   1),
               ("shear_taper=0", 0, 0, 0,  1,  1,   1),
               ("LAMBDA=0",      1,  1, 1,  0,  1,   1),
               ("kg=0",          1,  1, 1,  1,  0,   1),
               ("LAMBDA=0,kg=0", 1,  1, 1,  0,  0,   1),
               ("ALL_taper=0",   0,  0, 0,  0,  0,   1),
               ("G x0.01",       1,  1, 1,  1,  1, 0.01),
               ("G x100",        1,  1, 1,  1,  1, 100)]
    for mat in ("iso", "m45"):
        f = os.path.join(BENCH, "taper_square_solid_%s.npz" % mat)
        b = np.load(f, allow_pickle=True)
        tg = ts.tag_of(regime, mat, aR)
        So = 0.5 * (b["%s_seg" % tg] + b["%s_seg" % tg].T)
        SoL = 0.5 * (b["%s_L" % tg] + b["%s_L" % tg].T)
        print("\n### ablation %s : SEG C33 solid=%.4e C22 solid=%.4e (x1e9) ###"
              % (tg, So[2, 2] / 1e9, So[1, 1] / 1e9))
        print("%-16s %10s %10s %10s %12s | %10s" % ("config", "SEGc33%", "SEGc22%", "SEGc44%", "SEGc33(1e9)", "ringLc33%"))
        for nm, bge, bgl, y1, lam, kg, gsc in configs:
            _set(seg, bge, bgl, y1, lam, kg, gsc)
            rL, S6, rR = ts.shell_solve(tg, shear="mitc4_both", mesh_dir=MESH, res_dir=RES)
            S6 = 0.5 * (S6 + S6.T); rL = 0.5 * (rL + rL.T)
            def er(M, Mo, i):
                return 100 * (M[i, i] - Mo[i, i]) / Mo[i, i] if Mo[i, i] else float("nan")
            print("%-16s %+9.1f%% %+9.1f%% %+9.1f%% %12.4e | %+9.1f%%"
                  % (nm, er(S6, So, 2), er(S6, So, 1), er(S6, So, 3), S6[2, 2] / 1e9, er(rL, SoL, 2)))
        _set(seg)


def floorsweep(regime="thin", aR=0.7):
    """Sweep the drilling-denominator floor FLOOR33 (C33=n.b3 cap on the y3~0 GA3
    walls).  If GA3 is generated through the Lambda/drilling mechanism whose C33 is
    floored on those walls, the floor value should move C33 (and GJ=C44)."""
    import taper_study as ts
    import segment_element_general as seg
    MESH = os.path.join(HERE, "out", "taper_square", "meshes")
    RES = os.path.join(HERE, "out", "taper_square", "results")
    for mat in ("iso", "m45"):
        f = os.path.join(BENCH, "taper_square_solid_%s.npz" % mat)
        b = np.load(f, allow_pickle=True)
        tg = ts.tag_of(regime, mat, aR)
        So = 0.5 * (b["%s_seg" % tg] + b["%s_seg" % tg].T)
        SoL = 0.5 * (b["%s_L" % tg] + b["%s_L" % tg].T)
        print("\n### FLOOR33 sweep %s : C33 solid=%.4e C44(GJ) solid=%.4e (x1e9) ###"
              % (tg, So[2, 2] / 1e9, So[3, 3] / 1e9))
        print("%-8s %10s %10s %10s %12s | %10s" % ("FLOOR33", "C33%", "C22%", "C44%", "C44(1e9)", "ringLc33%"))
        for fl in (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005):
            seg.FLOOR33 = fl
            rL, S6, rR = ts.shell_solve(tg, shear="mitc4_both", mesh_dir=MESH, res_dir=RES)
            S6 = 0.5 * (S6 + S6.T); rL = 0.5 * (rL + rL.T)
            def er(M, Mo, i):
                return 100 * (M[i, i] - Mo[i, i]) / Mo[i, i] if Mo[i, i] else float("nan")
            print("%-8.3f %+9.1f%% %+9.1f%% %+9.1f%% %12.4e | %+9.1f%%"
                  % (fl, er(S6, So, 2), er(S6, So, 1), er(S6, So, 3), S6[3, 3] / 1e9, er(rL, SoL, 2)))
        seg.FLOOR33 = 5e-2


def circle_cmp(aRs=(1.0, 0.9, 0.7)):
    """CIRCLE tube (taper_study): C33 = n.b3 crosses 0 only at isolated points (not
    whole walls).  If the circle thin-taper GA3 is CLEAN, the square deficit is the
    whole-wall C33==0 degeneracy; if the circle is ALSO deficient, it is a general
    taper-shell-model limit."""
    import taper_study as ts
    MESH = os.path.join(HERE, "out", "taper_study", "meshes")
    RES = os.path.join(HERE, "out", "taper_study", "results")
    for mat in ("iso", "m45"):
        f = os.path.join(BENCH, "taper_study_solid_%s.npz" % mat)
        b = np.load(f, allow_pickle=True)
        print("\n### CIRCLE tube %s : SEG C33 & C22 err vs 3-D solid ###" % mat)
        print("%-6s %10s %10s %10s %10s" % ("aR", "C33_err%", "C22_err%", "C36_err%", "C44_err%"))
        for aR in aRs:
            tg = ts.tag_of("thin", mat, aR)
            key = "%s_seg" % tg
            if key not in b.files:
                print("  (%s missing in npz: %s)" % (key, list(b.files)[:3])); continue
            rL, S6, rR = ts.shell_solve(tg, shear="mitc4_both", mesh_dir=MESH, res_dir=RES)
            S6 = 0.5 * (S6 + S6.T); So = 0.5 * (b[key] + b[key].T)
            def er(i, j=None):
                j = i if j is None else j
                return 100 * (S6[i, j] - So[i, j]) / So[i, j] if So[i, j] else float("nan")
            print("%-6.2f %+9.1f%% %+9.1f%% %+9.1f%% %+9.1f%%" % (aR, er(2), er(1), er(2, 5), er(3)))


def dropsweep(regime="thin", aR=0.7):
    """DRILL_TOL sweep: drop the omega_3 drilling on |C33|<tol flat walls (instead of
    flooring 1/C33).  tol=0 is production baseline.  Want C33 -> solid without breaking
    C22/GJ/rings."""
    import taper_study as ts
    import segment_element_general as seg
    MESH = os.path.join(HERE, "out", "taper_square", "meshes")
    RES = os.path.join(HERE, "out", "taper_square", "results")
    for mat in ("iso", "m45"):
        f = os.path.join(BENCH, "taper_square_solid_%s.npz" % mat)
        b = np.load(f, allow_pickle=True)
        tg = ts.tag_of(regime, mat, aR)
        So = 0.5 * (b["%s_seg" % tg] + b["%s_seg" % tg].T)
        SoL = 0.5 * (b["%s_L" % tg] + b["%s_L" % tg].T)
        print("\n### DRILL_TOL sweep %s : C33 solid=%.4e GJ solid=%.4e (x1e9) ###"
              % (tg, So[2, 2] / 1e9, So[3, 3] / 1e9))
        print("%-9s %10s %10s %10s %10s | %10s" % ("DRILL_TOL", "C33%", "C22%", "GJ%", "C36%", "ringLc33%"))
        for tol in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            seg.DRILL_TOL = tol
            rL, S6, rR = ts.shell_solve(tg, shear="mitc4_both", mesh_dir=MESH, res_dir=RES)
            S6 = 0.5 * (S6 + S6.T); rL = 0.5 * (rL + rL.T)
            def er(M, Mo, i, j=None):
                j = i if j is None else j
                return 100 * (M[i, j] - Mo[i, j]) / Mo[i, j] if Mo[i, j] else float("nan")
            print("%-9.2f %+9.1f%% %+9.1f%% %+9.1f%% %+9.1f%% | %+9.1f%%"
                  % (tol, er(S6, So, 2), er(S6, So, 1), er(S6, So, 3), er(S6, So, 2, 5), er(rL, SoL, 2)))
        seg.DRILL_TOL = 0.0


def _rot6(beta):
    """6x6 beam-frame rotation about the axis by beta (order [ext,sh2,sh3,tor,b2,b3]).
    Shear pair (1,2) and bending pair (4,5) rotate as vectors; ext(0) & torsion(3) fixed."""
    c, s = math.cos(beta), math.sin(beta)
    R = np.eye(6)
    for i, j in ((1, 2), (4, 5)):
        R[i, i] = c; R[i, j] = -s; R[j, i] = s; R[j, j] = c
    return R


def _rotate_yaml(src_yaml, dst_yaml, beta, ax=2):
    """Rotate a shell mesh (nodes + per-element e1/e2/e3 orientations) by beta about
    axis `ax`, so the flat-wall normals get a nonzero C33=n.b3 (no drilling degeneracy)."""
    d = yaml.safe_load(open(src_yaml))
    cr = [j for j in range(3) if j != ax]                 # the two cross indices
    c, s = math.cos(beta), math.sin(beta)

    def rot(v):
        v = list(v); a, b = v[cr[0]], v[cr[1]]
        v[cr[0]] = c * a - s * b; v[cr[1]] = s * a + c * b
        return v
    d["nodes"] = [rot([float(x) for x in n]) for n in d["nodes"]]
    oris = []
    for o in d["elementOrientations"]:
        o = [float(x) for x in o]
        oris.append(rot(o[0:3]) + rot(o[3:6]) + rot(o[6:9]))
    d["elementOrientations"] = oris
    yaml.safe_dump(d, open(dst_yaml, "w"), default_flow_style=None, sort_keys=False)


def rotate_test(regime="thin", aR=0.7, betas_deg=(0, 15, 30, 45)):
    """Compute the square 6x6 in a frame rotated by beta about the beam axis (so no
    flat wall has C33=0), then rotate the 6x6 back and compare to the (original-frame)
    solid.  EA & GJ are frame-invariant -> they verify the transform; GA2/GA3 reveal
    whether removing the degeneracy recovers GA3."""
    import taper_study as ts
    MESH = os.path.join(HERE, "out", "taper_square", "meshes")
    ROT = os.path.join(HERE, "out", "taper_square", "rot_meshes")
    RES = os.path.join(HERE, "out", "taper_square", "results")
    os.makedirs(ROT, exist_ok=True)
    for mat in ("iso", "m45"):
        b = np.load(os.path.join(BENCH, "taper_square_solid_%s.npz" % mat), allow_pickle=True)
        tg = ts.tag_of(regime, mat, aR)
        So = 0.5 * (b["%s_seg" % tg] + b["%s_seg" % tg].T)
        print("\n### FRAME-ROTATION test %s : C33 solid=%.4e (x1e9) ###" % (tg, So[2, 2] / 1e9))
        print("%-7s %10s %10s %10s %10s %10s %10s" % ("beta", "EA%", "GA2%", "GA3%", "GJ%", "EI2%", "EI3%"))
        for bd in betas_deg:
            beta = math.radians(bd)
            rtag = "%s_rot%02d" % (tg, bd)
            _rotate_yaml(os.path.join(MESH, "shell_%s.yaml" % tg),
                         os.path.join(ROT, "shell_%s.yaml" % rtag), beta, ax=2)
            _, S6r, _ = ts.shell_solve(rtag, shear="mitc4_both", mesh_dir=ROT, res_dir=RES)
            R = _rot6(beta)
            S6 = R.T @ (0.5 * (S6r + S6r.T)) @ R                # back to original frame
            def er(i):
                return 100 * (S6[i, i] - So[i, i]) / So[i, i] if So[i, i] else float("nan")
            print("%-7d %+9.1f%% %+9.1f%% %+9.1f%% %+9.1f%% %+9.1f%% %+9.1f%%"
                  % (bd, er(0), er(1), er(2), er(3), er(4), er(5)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "peek"
    if cmd == "peek":
        peek()
    elif cmd == "base":
        base()
    elif cmd == "both":                 # thin AND thick, aR070
        base(regimes=("thin", "thick"), aRs=(0.7,))
    elif cmd == "refine":
        refine(sys.argv[2] if len(sys.argv) > 2 else "iso",
               sys.argv[3] if len(sys.argv) > 3 else "thin")
    elif cmd == "arsweep":
        arsweep(sys.argv[2] if len(sys.argv) > 2 else "iso")
        arsweep(sys.argv[2] if len(sys.argv) > 2 else "iso", regime="thick")
    elif cmd == "eps":
        epssweep()
    elif cmd == "ablate":
        ablate()
    elif cmd == "floor":
        floorsweep()
    elif cmd == "circle":
        circle_cmp()
    elif cmd == "drop":
        dropsweep()
    elif cmd == "rotate":
        rotate_test()
