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
    """Settled 6-DOF shear scheme.

    Tapered segment: FULL integration at every thickness -- the independent-omega3 element is
    locking-free under full integration, and assumed-strain (MITC) tying aliases the
    drilling-carried shear on flat walls (square thin taper collapses to -47% under mitc).

    Boundary ring: gamma23-tie (mitc4_g23) for thin walls, full for thick.  Circle/square are
    indifferent to the ring scheme; on the webbed multi-cell ring the gamma23-tie is the better
    choice for thin walls (GA2 -17% vs full's +29%).
    """
    if stage == "taper":
        return "full"
    return "mitc4_g23" if tR <= 0.02 + 1e-9 else "full"


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


def render_orientation(mesh_dir, tg, res_dir, title=""):
    """Render the actual computed shell mesh (from boundary_from_yaml.extract) in 3-D --
    quad edges coloured by subdomain (skin grey, web crimson) with subsampled e2 (blue) and
    e3 (black) orientation arrows. Returns a matplotlib Figure for inline display."""
    import io
    import contextlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from boundary_from_yaml import extract
    os.makedirs(res_dir, exist_ok=True)
    npz = os.path.join(res_dir, "shell_%s.npz" % tg)
    with contextlib.redirect_stdout(io.StringIO()):
        extract(os.path.join(mesh_dir, "shell_%s.yaml" % tg), npz)
    b = np.load(npz, allow_pickle=True)
    X = np.asarray(b["seg_x"]); Q = np.asarray(b["seg_cells"])
    e2 = np.asarray(b["seg_e2"]); e3 = np.asarray(b["seg_e3"]); sd = np.asarray(b["seg_subdom"])
    cen = X[Q].mean(1)
    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    web = sd != sd.min()
    for q, w in zip(Q, web):
        loop = list(q) + [int(q[0])]
        ax.plot(X[loop, 2], X[loop, 0], X[loop, 1], color=("crimson" if w else "0.75"), lw=0.35)
    L = 0.09 * float((X.max(0) - X.min(0)).max())
    st = max(1, len(Q) // 180)
    for k in range(0, len(Q), st):
        c = cen[k]
        ax.quiver(c[2], c[0], c[1], e2[k, 2], e2[k, 0], e2[k, 1], length=L, color="blue", lw=0.7)
        ax.quiver(c[2], c[0], c[1], e3[k, 2], e3[k, 0], e3[k, 1], length=L, color="black", lw=0.7)
    ax.set_xlabel("span z"); ax.set_ylabel("x"); ax.set_zlabel("y")
    ax.set_title(title)
    try:
        ax.set_box_aspect((3, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=18, azim=-70)
    fig.tight_layout()
    png = os.path.join(res_dir, "orient_%s.png" % tg)
    fig.savefig(png, dpi=95, bbox_inches="tight")
    plt.close(fig)
    return png


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
