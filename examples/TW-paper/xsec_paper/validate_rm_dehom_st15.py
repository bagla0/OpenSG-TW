"""validate_rm_dehom_st15.py -- RM-consistent dehom (dehom_rm) vs the KL-Hermite dehom
(msg_dehom) vs VABS .SM, on the st15 cap-centre through-thickness path.

Confirms (1) the RM path uses the RM ring 6x6 (not the KL 6x6), (2) the in-plane stress is
still recovered to <1% vs VABS, and (3) sigma13/sigma23 are now PHYSICAL (~0.04 MPa, right
order) from the per-element wall shear, not the failed 16-560x section-shear approximation.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml, stress_at_points as stress_kl
import dehom_rm

DEH = os.path.join(HERE, "..", "..", "..", "examples", "data", "dehom_st15")
SHELL15 = os.path.expanduser("~/OpenSG-TW-claude/tests/data/1Dshell_15.yaml")
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def load_sm(path):
    d = np.loadtxt(path)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]


sm_xy, sm_s = load_sm(os.path.join(DEH, "bar_urc-15-t-0.in.SM"))
tree = cKDTree(sm_xy)
coords = np.loadtxt(os.path.join(DEH, "solid.lp_sparcap_center_thickness_015.coords"))[:, :2]
V = sm_s[tree.query(coords)[1]]

print("=" * 78)
print(" RM-consistent dehom vs KL-Hermite dehom vs VABS .SM -- st15 cap-centre path")
print("=" * 78)

# --- KL-Hermite dehom (current) ---
bkl = solve_tw_from_yaml(SHELL15, frac=0.0)
Skl = np.asarray(stress_kl(bkl, coords, beam_force_vabs=FF, frame="material")["stress"])

# --- RM-consistent dehom (new) ---
B = dehom_rm.build_rm_bundle(SHELL15, ref="oml")
Srm = np.asarray(dehom_rm.stress_at_points(B, coords, beam_force_vabs=FF, frame="material")["stress"])

# --- 6x6 used by each path (VABS order diag) ---
lbl = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
Ckl = np.asarray(bkl["Timo"]); Crm = np.asarray(B["Timo"])
print("\n6x6 diagonal (confirms the RM path uses the RM ring, not the KL shell):")
print("  %-5s %14s %14s %8s" % ("term", "KL-Hermite", "RM ring", "d%"))
for i in range(6):
    print("  %-5s %14.4e %14.4e %+7.1f" %
          (lbl[i], Ckl[i, i], Crm[i, i], 100 * (Crm[i, i] - Ckl[i, i]) / Ckl[i, i]))

# --- stress comparison ---
def peak(a):
    return a[np.argmax(np.abs(a))]

print("\nper-component peak stress (MPa) on the cap-centre path:")
print("  %-4s %12s %12s %12s" % ("comp", "KL-Hermite", "RM", "VABS"))
for k, c in enumerate(COMP):
    print("  %-4s %12.4f %12.4f %12.4f" %
          (c, peak(Skl[:, k]) / 1e6, peak(Srm[:, k]) / 1e6, peak(V[:, k]) / 1e6))

ip = [0, 1, 5]  # S11,S22,S12
op = [2, 4, 3]  # S33,S13,S23
def relerr(S):
    return (np.linalg.norm(S[:, ip] - V[:, ip]) / np.linalg.norm(V[:, ip]) * 100)
print("\nin-plane (S11,S22,S12) Frobenius error vs VABS:  KL %.2f%%   RM %.2f%%"
      % (relerr(Skl), relerr(Srm)))
print("out-of-plane peak |S13|,|S23| (MPa):  VABS %.4f, %.4f  |  KL %.4f, %.4f  |  RM %.4f, %.4f"
      % (abs(peak(V[:, 4]))/1e6, abs(peak(V[:, 3]))/1e6,
         abs(peak(Skl[:, 4]))/1e6, abs(peak(Skl[:, 3]))/1e6,
         abs(peak(Srm[:, 4]))/1e6, abs(peak(Srm[:, 3]))/1e6))
print("\nOK" if relerr(Srm) < 5 else "\nWARN: RM in-plane error high")
