"""
RM 2-step dehomogenization with NON-ZERO transverse-shear 3D stress, recovered
from the V1 shear-warping (so the shear lands in the webs/junctions, as VABS).

Step 1: FF -> beam strain st=inv(C6)@FF -> macro recovery (st_m, st_cl1, st_cl2);
        warping  a = V0@st_m + V1@st_cl1 ; recovered shell transverse-shear strain
        Gamma_G(s) = BGq(s) @ a  (= the RM dehom step-1, the piece V1 enables).
Step 2: in-plane (S11,S22,S33,S12) from the existing MSG dehom; transverse shear
        sigma13(z) = (F13/h) Gamma_G[0] * gx(z)/<gx>  (parabolic, 0 at the faces),
        sigma23(z) likewise.

debug_distribution() prints max|Gamma_G| per layup -- it must be large on the
WEBS (which carry the transverse shear) and ~0 on the caps, matching VABS.

Outputs (material frame): outputs/rm_dehom/<path>_rm.png and _rm_vs_kf.png.
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, solve_tw_from_yaml, stress_at_points, compute_ABD_matrix
from fe_jax.msg_mesh import read_mesh, mesh_curvature
from fe_jax.msg_dehom import _macro_recovery
from msg_rm import _lagrange
from msg_rm_timo import _elem_BD_BG_BL, timoshenko_rm
from transverse_shear import transverse_shear_stiffness

YAML = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
PDIR = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data"
        r"\opensg-FEniCS\data\st15_path_coords-20260614T203452Z-3-001\st15_path_coords")
SM = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data"
      r"\opensg-FEniCS\data\bar_urc-15-t-0.in.SM")
OUT = os.path.join(HERE, "..", "outputs", "rm_dehom")
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def load_sm():
    d = np.loadtxt(SM); return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]


def project_rm(nodes, elems, P):
    best = (np.inf, 0, 0.0)
    for e, el in enumerate(elems):
        A, B = nodes[el[0]], nodes[el[-1]]; AB = B-A; L2 = float(AB@AB)
        t = 0.0 if L2 < 1e-30 else float(np.clip((P-A)@AB/L2, 0, 1))
        d = float(np.hypot(*(P-(A+t*AB))))
        if d < best[0]: best = (d, e, t)
    return best[1], 2*best[2]-1.0                      # element, xi in [-1,1]


def recover_GG_field(rm, FF, eval_pts=None):
    """Recovered shell transverse-shear strain Gamma_G=[2g13,2g23] from V0,V1.
    If eval_pts is None, evaluate at element midpoints (for the debug)."""
    nodes, elems, lpe, k22 = rm["nodes"], rm["elems"], rm["lpe"], rm["k22"]
    V0, V1, C6, p = rm["V0"], rm["V1"], rm["C6"], rm["p"]
    st = np.linalg.solve(C6, FF)
    _, st_m, st_cl1, st_cl2 = _macro_recovery(C6, st)
    a = V0 @ st_m + V1 @ st_cl1                          # eps_h warping for Gamma_G
    nodes_xi = _lagrange(p)
    pts = eval_pts if eval_pts is not None else \
        np.array([nodes[el].mean(0) for el in elems])
    GG = np.zeros((len(pts), 2)); el_of = np.zeros(len(pts), int)
    for ip, P in enumerate(pts):
        e, xi = project_rm(nodes, elems, P)
        X = nodes[elems[e]]
        _, BGq, _, _ = _elem_BD_BG_BL(nodes_xi, xi, X, None, float(k22[e]), p)
        g = np.concatenate([[5*n, 5*n+1, 5*n+2, 5*n+3, 5*n+4] for n in elems[e]])
        GG[ip] = BGq @ a[g]; el_of[ip] = e
    return GG, el_of


def debug_distribution(rm):
    GG, el_of = recover_GG_field(rm, FF)
    lpe = rm["lpe"]
    print("=== DEHOM DEBUG: recovered |Gamma_G| (shell transverse shear) by layup ===")
    print("    (must be LARGE on webs, ~0 on caps -- matching VABS S13)")
    by = {}
    for i, e in enumerate(el_of):
        by.setdefault(lpe[e], []).append(np.hypot(GG[i, 0], GG[i, 1]))
    for ln in sorted(by, key=lambda k: -max(by[k])):
        print(f"    {ln:10s} max|Gamma_G| = {max(by[ln]):.3e}  (n={len(by[ln])})")


def transverse_shear_at(rm, coords, G_by, h_by, shapes_by):
    GG, el_of = recover_GG_field(rm, FF, eval_pts=coords)
    nodes, elems, lpe = rm["nodes"], rm["elems"], rm["lpe"]
    S13 = np.zeros(len(coords)); S23 = np.zeros(len(coords))
    for ip, P in enumerate(coords):
        e = el_of[ip]; ln = lpe[e]; Gf = G_by[ln]; h = h_by[ln]
        zs, gxn, gyn, mx, my = shapes_by[ln]
        # inward depth of the point
        A, B = nodes[elems[e][0]], nodes[elems[e][-1]]; t = (B-A)/np.hypot(*(B-A))
        n = np.array([t[1], -t[0]]); cen = nodes.mean(0)
        prj = A + np.clip((P-A)@(B-A)/((B-A)@(B-A)), 0, 1)*(B-A)
        if (cen-prj)@n < 0: n = -n
        z = float((P-prj)@n)
        sh13 = np.interp(np.clip(z, 0, h), zs, gxn)/mx
        sh23 = np.interp(np.clip(z, 0, h), zs, gyn)/my
        S13[ip] = Gf[0, 0]/h * GG[ip, 0] * sh13
        S23[ip] = Gf[1, 1]/h * GG[ip, 1] * sh23
    return S13, S23


def run(kb, rm, G_by, h_by, shapes_by, path_file, name, sm_xy, sm_s):
    coords = np.loadtxt(os.path.join(PDIR, path_file))[:, :2]
    z = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))]
    vabs = sm_s[cKDTree(sm_xy).query(coords)[1]]
    S = stress_at_points(kb, coords, beam_force_vabs=FF, frame="material")["stress"].copy()
    S13, S23 = transverse_shear_at(rm, coords, G_by, h_by, shapes_by)
    S_rm = S.copy(); S_rm[:, 4] = S13; S_rm[:, 3] = S23     # cols S23=3, S13=4
    os.makedirs(OUT, exist_ok=True)
    _panel(os.path.join(OUT, f"{name}_rm.png"),
           f"Station 15 {name} -- RM 2-step dehom (V1 transverse shear) vs VABS",
           z, [("MSG-TW RM", S_rm, "r-o"), ("VABS (.SM)", vabs, "g--^")])
    _panel(os.path.join(OUT, f"{name}_rm_vs_kf.png"),
           f"Station 15 {name} -- RM vs Kirchhoff vs VABS",
           z, [("MSG-TW RM", S_rm, "r-o"), ("MSG-TW Kirchhoff", S, "b:s"),
               ("VABS (.SM)", vabs, "g--^")])
    print(f"{name}: max|S13| RM {np.max(np.abs(S13))/1e6:.2f}  VABS "
          f"{np.max(np.abs(vabs[:,4]))/1e6:.2f} MPa (Kirchhoff 0)")


def _panel(fname, title, z, series):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5)); fig.suptitle(title, fontweight="bold")
    for j, c in enumerate(COMP):
        a = ax.flat[j]; oop = c in ("S33", "S13", "S23")
        for lbl, Sd, fmt in series:
            a.plot(z*1e3, Sd[:, j]/1e6, fmt, ms=3.5, label=lbl)
        a.set_title(f"$\\sigma_{{{c[1:]}}}$"+("  [out-of-plane]" if oop else ""),
                    fontweight="bold", color=("darkred" if oop else "black"))
        a.set_xlabel("path (mm)"); a.set_ylabel(f"{c} (MPa)")
        a.grid(True, ls=":", alpha=0.6); a.legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(fname, dpi=150); plt.close(fig)
    print("wrote", fname)


def main():
    kb = solve_tw_from_yaml(YAML, frac=0.0)                # Kirchhoff: in-plane + projection
    n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = np.asarray(mesh_curvature(nodes, cells, elements, is_closed=False))
    D_by = {ln: np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
            for ln, i in layup_db.items()}
    G_by, h_by, shapes_by = {}, {}, {}
    for ln, i in layup_db.items():
        Gm, _, (zs, gxn, gyn) = transverse_shear_stiffness(
            i["thick"], i["angles"], i["mat_names"], mat_db)
        h = float(sum(i["thick"])); G_by[ln] = Gm; h_by[ln] = h
        shapes_by[ln] = (zs, gxn, gyn, np.mean(gxn), np.mean(gyn))
    C6, _, V0, V1 = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1, return_warp=True)
    rm = {"nodes": nodes2d, "elems": elems, "lpe": lpe, "k22": k22,
          "V0": V0, "V1": V1, "C6": C6, "p": 1}

    debug_distribution(rm)
    sm_xy, sm_s = load_sm()
    for pf, nm in [("solid.circumferential_015.coords", "circumferential"),
                   ("solid.lp_sparcap_center_thickness_015.coords", "sparcap_center"),
                   ("solid.lp_sparcap_left_edge_thickness_015.coords", "leftspar")]:
        run(kb, rm, G_by, h_by, shapes_by, pf, nm, sm_xy, sm_s)


if __name__ == "__main__":
    main()
