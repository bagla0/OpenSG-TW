"""
RM 2-step dehomogenization with NON-ZERO transverse-shear 3D stress.

In-plane (S11,S22,S33,S12): the existing MSG in-plane dehom (RM ~ Kirchhoff).
Transverse shear (S13,S23): recovered by 3D equilibrium -- the piece RM enables
that the Kirchhoff-shell dehom gives as 0:

  d sigma_13/dz = - d sigma_11/dx1 ,   sigma_11 = Q11(z)( x3 k2 - x2 k3 + ... )
  d()/dx1 of the bending => the SHEAR-induced curvature gradients k2',k3'
  (moment gradients = the beam transverse shear forces F3,F2):
      [k2'; k3'] = inv([[EI2,C56],[C56,EI3]]) @ [M2'; M3'] ,  M2'=F3, M3'=-F2
  sigma_13(z) = - INT_0^z Q11(z')[ (x3o+z' n3) k2' - (x2o+z' n2) k3' ] dz'
(parabolic, zero at the OML face; the through-thickness g(z) shear flow).

Outputs (material frame), for the circumferential / spar-cap-centre / left-edge paths:
  outputs/rm_dehom/<path>_rm.png         RM (S13,S23 != 0) vs VABS .SM
  outputs/rm_dehom/<path>_rm_vs_kf.png   + Kirchhoff overlay (S13=S23=0)

STATUS / CAVEAT: this demonstrates the RM *capability* -- non-zero S13/S23 where
the Kirchhoff-shell dehom gives exactly 0.  The transverse-shear MAGNITUDE here
uses a LEADING-ORDER recovery (uniform section shear strain projected onto each
wall, parabolic through-thickness).  That mis-distributes the shear: VABS shows
S13 large at the cap/web JUNCTIONS (the webs carry the transverse shear) and ~0
in the cap centre, whereas the uniform-strain projection spreads it onto the
caps.  The physically-correct distribution requires the V1 shear-warping (which
wall carries the shear flow) -- the final step, not yet wired into the recovery.
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import solve_tw_from_yaml, stress_at_points
from fe_jax.msg_dehom import _project_point
from transverse_shear import _ply_Q_and_G

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


def transverse_shear_path(bundle, coords, g_beam, G_by, h_by, shapes_by, layup_db):
    """Wall transverse shear: project the recovered beam transverse-shear strain
    g_beam=[2g12,2g13] onto the wall (n,t), apply G_eff=F_FSDT/h, and distribute
    through the thickness with the PARABOLIC g(z) shear flow (zero at the faces):
      sigma13(z) = (F13/h)(g_beam . n) * gx(z)/<gx>
      sigma23(z) = (F23/h)(g_beam . t) * gy(z)/<gy>
    """
    corners = np.asarray(bundle["corners"]); rc = np.asarray(bundle["red_cells"])
    xd2 = np.asarray(bundle["xd2"]); xd3 = np.asarray(bundle["xd3"])
    cen = corners.mean(0); lpe = bundle["layup_per_elem"]
    S13 = np.zeros(len(coords)); S23 = np.zeros(len(coords))
    for ip, p in enumerate(coords):
        e, xi, pr = _project_point(corners, rc, p)
        t2, t3 = float(xd2[e]), float(xd3[e]); n2, n3 = t3, -t2
        mid = 0.5*(corners[int(rc[e, 0])]+corners[int(rc[e, 1])])
        if (cen[0]-mid[0])*n2 + (cen[1]-mid[1])*n3 < 0: n2, n3 = -n2, -n3
        z = float((p[0]-pr[0])*n2 + (p[1]-pr[1])*n3)            # inward depth
        ln = lpe[e]; Gf = G_by[ln]; h = h_by[ln]
        zs, gxn, gyn, mx, my = shapes_by[ln]
        gn = g_beam[0]*n2 + g_beam[1]*n3
        gt = g_beam[0]*t2 + g_beam[1]*t3
        sh13 = np.interp(np.clip(z, 0, h), zs, gxn) / mx        # avg-normalized, 0 at faces
        sh23 = np.interp(np.clip(z, 0, h), zs, gyn) / my
        S13[ip] = Gf[0, 0]/h * gn * sh13
        S23[ip] = Gf[1, 1]/h * gt * sh23
    return S13, S23


def run(bundle, layup_db, G_by, h_by, shapes_by, path_file, name, sm_xy, sm_s):
    coords = np.loadtxt(os.path.join(PDIR, path_file))[:, :2]
    z = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))]
    vabs = sm_s[cKDTree(sm_xy).query(coords)[1]]
    out = stress_at_points(bundle, coords, beam_force_vabs=FF, frame="material")
    S = out["stress"].copy()                                    # Kirchhoff: S13=S23=0
    # recovered beam transverse shear strains [2g12, 2g13] = (inv Timo @ FF)[1,2]
    g_beam = np.linalg.solve(np.asarray(bundle["Timo"]), FF)[[1, 2]]
    S13, S23 = transverse_shear_path(bundle, coords, g_beam, G_by, h_by, shapes_by, layup_db)
    S_rm = S.copy(); S_rm[:, 4] = S13; S_rm[:, 3] = S23         # cols: S23=3, S13=4

    os.makedirs(OUT, exist_ok=True)
    # plot 1: RM vs VABS
    _panel(os.path.join(OUT, f"{name}_rm.png"),
           f"Station 15 {name} -- RM 2-step dehom (S13,S23 != 0) vs VABS",
           z, [("MSG-TW RM", S_rm, "r-o"), ("VABS (.SM)", vabs, "g--^")])
    # plot 2: RM + Kirchhoff overlay
    _panel(os.path.join(OUT, f"{name}_rm_vs_kf.png"),
           f"Station 15 {name} -- RM vs Kirchhoff vs VABS (note S13/S23)",
           z, [("MSG-TW RM", S_rm, "r-o"), ("MSG-TW Kirchhoff", S, "b:s"),
               ("VABS (.SM)", vabs, "g--^")])
    print(f"{name}: max|S13| RM {np.max(np.abs(S13))/1e6:.2f} MPa  "
          f"VABS {np.max(np.abs(vabs[:,4]))/1e6:.2f} MPa  (Kirchhoff 0)")


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
    from fe_jax import load_yaml
    from transverse_shear import transverse_shear_stiffness
    bundle = solve_tw_from_yaml(YAML, frac=0.0)
    _, _, mat_db, layup_db, _ = load_yaml(YAML)
    G_by, h_by, shapes_by = {}, {}, {}
    for ln, i in layup_db.items():
        Gm, _, (zs, gxn, gyn) = transverse_shear_stiffness(
            i["thick"], i["angles"], i["mat_names"], mat_db)
        h = float(sum(i["thick"]))
        G_by[ln] = Gm; h_by[ln] = h
        shapes_by[ln] = (zs, gxn, gyn, np.mean(gxn), np.mean(gyn))
    sm_xy, sm_s = load_sm()
    for pf, nm in [("solid.circumferential_015.coords", "circumferential"),
                   ("solid.lp_sparcap_center_thickness_015.coords", "sparcap_center"),
                   ("solid.lp_sparcap_left_edge_thickness_015.coords", "leftspar")]:
        run(bundle, layup_db, G_by, h_by, shapes_by, pf, nm, sm_xy, sm_s)


if __name__ == "__main__":
    main()
