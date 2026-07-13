"""Debug the LP spar-cap through-thickness DISPLACEMENT: RM disp_at_points returns only the
mid-surface (OML) warping, constant through the wall, while VABS varies with depth.  Test adding
the RM director term u(z) = u_mid + z*(omega x e3) (both signs) against VABS .U along the cap."""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm
from opensg_jax.fe_jax.msg_dehom import _project_point

D2 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "data", "2d_yaml"))
SHELL = os.path.abspath(os.path.join(HERE, "..", "..", "..", "examples", "TW-paper",
                                     "iea22_blade", "data", "shell_r020.yaml"))
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])
cap = np.loadtxt(os.path.join(D2, "solid.lp_sparcap_right_thickness_r020.coords"))[:, :2]
U = np.loadtxt(os.path.join(D2, "iea_r020.sg.U"))
utree = cKDTree(U[:, 1:3])
B = dehom_rm.build_rm_bundle(SHELL, ref="oml")
st, st_m, aA, aB = dehom_rm._macro_fields(B, FF, None)
wn = np.asarray(aA).reshape(-1, 6)
corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)

print(" dep(mm)  z(mm) | VABS u1  mid u1  +dir u1  -dir u1 | VABS u3  +dir u3  -dir u3")
for p in cap:
    e, xi, pr = _project_point(corners, rc, p)
    c0, c1 = int(rc[e, 0]), int(rc[e, 1])
    umid = (1 - xi) * wn[c0, 0:3] + xi * wn[c1, 0:3]
    om = (1 - xi) * wn[c0, 3:6] + xi * wn[c1, 3:6]
    t2, t3 = corners[c1] - corners[c0]; tl = np.hypot(t2, t3); t2, t3 = t2 / tl, t3 / tl
    n2, n3 = t3, -t2
    if (cen[0] - pr[0]) * n2 + (cen[1] - pr[1]) * n3 < 0:
        n2, n3 = -n2, -n3
    z = (p[0] - pr[0]) * n2 + (p[1] - pr[1]) * n3          # depth from OML along inward normal
    e3 = np.array([0.0, n2, n3])
    dvec = np.cross(om, e3) * z                            # director term
    up = (umid + dvec) * 1e3; um = (umid - dvec) * 1e3; u0 = umid * 1e3
    V = U[utree.query(p)[1], 3:6] * 1e3
    dep = np.hypot(p[0] - cap[0, 0], p[1] - cap[0, 1]) * 1e3
    print(" %6.1f  %6.2f | %7.3f %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f"
          % (dep, z * 1e3, V[0], u0[0], up[0], um[0], V[2], up[2], um[2]))
