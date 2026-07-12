"""dehom_st15_rm8x8.py -- RM 8x8 transverse-shear recovery, VALIDATION.
Recovers sigma13,sigma23 through the wall from the SAME complementary-energy shear-flow
shapes ghat(z) that build the 2x2 G (transverse_shear_stiffness), driven -- as the
reviewer-sanctioned LEADING-ORDER resultant -- by the section transverse-shear force
[Q1,Q2] = G_2x2 @ [2g13,2g23]_section, with the section engineering shear strains taken
from the beam compliance st = C_eff^-1 FF.  Compares sigma13/sigma23 to VABS .SM on the
cap-centre path.  Ships ONLY if it validates (sign, profile, order of magnitude).
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml, stress_at_points
from opensg_jax.fe_jax.msg_mesh import load_yaml
from opensg_jax.fe_jax.msg_transverse_shear import transverse_shear_stiffness

DEH = os.path.join(REPO, "examples", "data", "dehom_st15")
SHELL15 = os.path.join(REPO, "examples", "data", "1d_yaml", "st15_shell.yaml")
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]


def load_sm(p):
    d = np.loadtxt(p)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]


# --- per-layup transverse-shear shear-flow shapes ghat(z) and the 2x2 G ---
_n, _e, matdb, layup_db, _e2l = load_yaml(SHELL15)
SHEAR = {}
for ln, info in layup_db.items():
    G, _rec, (zs, ghx, ghy) = transverse_shear_stiffness(
        info["thick"], info["angles"], info["mat_names"], matdb)
    SHEAR[ln] = dict(G=np.asarray(G), zs=np.asarray(zs), ghx=np.asarray(ghx),
                     ghy=np.asarray(ghy), h=float(sum(info["thick"])))


def sig_shear(layup, z, Q):
    """sigma13, sigma23 at depth z from the shear-flow shapes and resultant Q=[Q1,Q2]."""
    s = SHEAR[layup]
    i = min(int(z / s["h"] * len(s["zs"])), len(s["zs"]) - 1)
    return s["ghx"][i] * Q[0], s["ghy"][i] * Q[1]


bundle = solve_tw_from_yaml(SHELL15, frac=0.0)
Ceff = np.asarray(bundle["Timo"])
st = np.linalg.inv(Ceff) @ FF                    # [e11, 2g12, 2g13, k1, k2, k3] (VABS order)
gsec = np.array([st[1], st[2]])                  # section engineering transverse shears
lay = list(bundle["layup_per_elem"])
elements = np.asarray(bundle["elements"])        # 1-based connectivity per element

sm_xy, sm_s = load_sm(os.path.join(DEH, "bar_urc-15-t-0.in.SM"))
tree = cKDTree(sm_xy)
coords = np.loadtxt(os.path.join(DEH, "solid.lp_sparcap_center_thickness_015.coords"))[:, :2]
out = stress_at_points(bundle, coords, beam_force_vabs=FF, frame="material")
S = np.asarray(out["stress"]); el = np.asarray(out["elem"]); dep = np.asarray(out["depth"]) \
    if "depth" in out else None
V = sm_s[tree.query(coords)[1]]

# recover sigma13/sigma23 at each cap-centre point (leading-order resultant Q = G @ gsec)
z_mm = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))] * 1e3
print("depth   S13_shell  S13_vabs   S23_shell  S23_vabs   (MPa)   [layup]")
S13 = np.zeros(len(coords)); S23 = np.zeros(len(coords))
for i in range(len(coords)):
    e = int(el[i]); ln = lay[e]
    Q = SHEAR[ln]["G"] @ gsec
    # depth of the query point within its wall: use the reported through-thickness pos
    zdep = dep[i] if dep is not None else SHEAR[ln]["h"] * 0.5
    s13, s23 = sig_shear(ln, zdep, Q)
    S13[i], S23[i] = s13, s23
    print("%6.1f  %9.4f %9.4f   %9.4f %9.4f     %s"
          % (z_mm[i], s13 / 1e6, V[i, 4] / 1e6, s23 / 1e6, V[i, 3] / 1e6, ln))
print("\npeak |S13| shell=%.4f  VABS=%.4f MPa | peak |S23| shell=%.4f  VABS=%.4f MPa"
      % (np.abs(S13).max()/1e6, np.abs(V[:,4]).max()/1e6,
         np.abs(S23).max()/1e6, np.abs(V[:,3]).max()/1e6))
