"""Shared helpers for the RM_taper examples (circle / square / webbed ellipse).

The 6-DOF independent-omega3 Reissner-Mindlin shell (the paper's formulation) is used for
BOTH the tapered segment and the boundary rings, with the transverse-shear scheme selected
by wall slenderness and stage:

    stage             thin  (t/R <= 0.02)     thick (t/R > 0.02)
    tapered segment   MITC   (mitc4_both)      full  (2x2 Gauss)
    boundary ring     gamma23-tie (mitc4_g23)  full

Each case loads a 1-D shell mesh YAML and a pre-computed conforming-solid 6x6 reference
(FEniCS 3-D solid), computes the equivalent-beam Timoshenko 6x6, and prints the per-term
%-error.  Input: YAML mesh.  Output: Timoshenko 6x6 (C^b).
"""
import os
import sys
import json
import numpy as np


def _find_root(start):
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "pyproject.toml")) and os.path.isdir(os.path.join(d, "mitc_rm_segment")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root (pyproject.toml + mitc_rm_segment) not found")


CC = _find_root(os.path.dirname(__file__))
for p in (CC, os.path.join(CC, "mitc_rm_segment"), os.path.join(CC, "opensg_jax")):
    if p not in sys.path:
        sys.path.insert(0, p)

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def shear_for(stage, tR):
    """Thin/thick shear rule (6-DOF everywhere)."""
    thin = tR <= 0.02 + 1e-9
    if stage == "taper":
        return "mitc4_both" if thin else "full"
    return "mitc4_g23" if thin else "full"          # boundary ring


def solve_boundary(mesh_dir, tg, res_dir, shear):
    """6-DOF boundary-ring Timoshenko 6x6 from the LEFT cross-section of a shell taper mesh."""
    from boundary_from_yaml import extract
    from run_ring_indep import ring_indep
    from segment_element import compute_k22
    from solve_segment_jax import _material_by_section
    import io
    import contextlib
    os.makedirs(res_dir, exist_ok=True)
    npz = os.path.join(res_dir, "shell_%s.npz" % tg)
    with contextlib.redirect_stdout(io.StringIO()):
        extract(os.path.join(mesh_dir, "shell_%s.yaml" % tg), npz)
    b = np.load(npz, allow_pickle=True)
    ax = int(b["axis"]); cross = tuple(j for j in range(3) if j != ax)
    D_by, G_by = _material_by_section(json.loads(str(b["sections"])),
                                      json.loads(str(b["materials"])), center_ref=True)
    rx = np.asarray(b["L_x"]); rc = np.asarray(b["L_cells"])
    rs = np.asarray(b["L_subdom"]); re3 = np.asarray(b["L_e3"])
    kr = compute_k22(rx[rc].mean(1), np.asarray(b["L_e2"]), re3, rc)
    C6 = ring_indep(rx, rc, rs, re3, D_by, G_by, kr, ax, list(cross), shear=shear)
    return np.asarray(C6)


def solve_taper(mesh_dir, tg, res_dir, shear):
    """6-DOF tapered-segment Timoshenko 6x6."""
    from run_indep import shell_solve_lagrange
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        S6 = shell_solve_lagrange(tg, mesh_dir, res_dir, shear=shear)
    return np.asarray(S6)


def report(title, So, Sh, shear):
    So = 0.5 * (So + So.T); Sh = 0.5 * (Sh + Sh.T)
    print("\n" + "-" * 66)
    print("%s   [6-DOF RM shell, shear=%s]" % (title, shear))
    # 1) the Timoshenko diagonal -- the headline 6x6
    print("  diagonal Timoshenko stiffness (x1e9):")
    print("  %-5s %13s %13s %9s" % ("term", "solid", "RM shell", "%err"))
    for i in range(6):
        so, sh = So[i, i], Sh[i, i]
        e = 100 * (sh - so) / so if so != 0 else float("nan")
        print("  %-5s %13.5f %13.5f %+8.1f%%" % (LBL[i], so / 1e9, sh / 1e9, e))
    # 2) the significant off-diagonal couplings
    thr = 1e-2 * abs(np.diag(So)).max()
    cpl = [(i, j) for i in range(6) for j in range(i + 1, 6)
           if abs(So[i, j]) > thr or abs(Sh[i, j]) > thr]
    if cpl:
        print("  couplings (x1e9):")
        for i, j in cpl:
            so, sh = So[i, j], Sh[i, j]
            e = 100 * (sh - so) / so if so != 0 else float("nan")
            flag = "  <-- sign flip" if so * sh < 0 else ""
            print("  C%d%d   %12.5f %12.5f %+8.1f%%%s" % (i + 1, j + 1, so / 1e9, sh / 1e9, e, flag))


def run_geometry(name, mesh_dir, res_dir, cases):
    """cases: list of (regime, tR, tg, solid_boun_6x6, solid_taper_6x6)."""
    print("=" * 66)
    print("RM-taper reproducible example:  %s  ([-45] ply, m45)" % name)
    print("=" * 66)
    for regime, tR, tg, sol_boun, sol_taper in cases:
        print("\n########  %s  (t/R = %.2f)  ########" % (regime.upper(), tR))
        sb = shear_for("boundary", tR); st = shear_for("taper", tR)
        Cb = solve_boundary(mesh_dir, tg, res_dir, sb)
        report("BOUNDARY ring vs solid boundary", np.asarray(sol_boun), Cb, sb)
        Ct = solve_taper(mesh_dir, tg, res_dir, st)
        report("TAPERED segment vs solid segment", np.asarray(sol_taper), Ct, st)
