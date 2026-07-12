"""Diagnose the st15 dehom in-plane mismatch: per-point RM vs VABS (S11,S22,S12) with the
projection element/xi/depth, to separate facesheet-interface + web-junction folds from real
recovery error."""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", ".."))); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm

DEH = os.path.join(HERE, "..", "..", "..", "examples", "data", "dehom_st15")
SHELL = os.path.expanduser("~/OpenSG-TW-claude/tests/data/1Dshell_15.yaml")
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])
d = np.loadtxt(os.path.join(DEH, "bar_urc-15-t-0.in.SM"))
sm_xy, sm_s = d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]
tree = cKDTree(sm_xy)
B = dehom_rm.build_rm_bundle(SHELL, ref="oml")


def diag(fn, tag, around):
    coords = np.loadtxt(os.path.join(DEH, fn))[:, :2]
    z = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))]
    R = dehom_rm.stress_at_points(B, coords, beam_force_vabs=FF, frame="material")
    S = np.asarray(R["stress"]); el = R["elem"]; dep = R["depth"]
    dist, idx = tree.query(coords)
    V = sm_s[idx]
    print("\n==== %s : %d pts ====" % (tag, len(coords)))
    print(" %5s %5s %5s %6s | %8s %8s %8s | %8s %8s %8s | %7s"
          % ("pt", "elem", "dep_mm", "match_mm", "S11_RM", "S11_V", "d%", "S12_RM", "S12_V", "S22_RM", "matchd"))
    ip = np.linalg.norm(S[:, [0, 1, 5]] - V[:, [0, 1, 5]]) / np.linalg.norm(V[:, [0, 1, 5]]) * 100
    for i in range(len(coords)):
        de = 100 * (S[i, 0] - V[i, 0]) / V[i, 0] if abs(V[i, 0]) > 1e3 else 0.0
        print(" %5d %5d %6.1f %8.1f | %8.2f %8.2f %+7.1f | %8.3f %8.3f %8.3f | %7.1f"
              % (i, el[i], dep[i]*1e3, (z[i]*(1 if around else 1e3)), S[i,0]/1e6, V[i,0]/1e6, de,
                 S[i,5]/1e6, V[i,5]/1e6, S[i,1]/1e6, dist[i]*1e3))
    print(" in-plane Frobenius = %.1f%% ; worst-S11 pts:" % ip)
    e11 = np.abs(S[:, 0] - V[:, 0])
    for i in np.argsort(e11)[-4:][::-1]:
        print("   pt %d  elem %d dep %.1fmm  RM %.1f VABS %.1f  match %.1fmm"
              % (i, el[i], dep[i]*1e3, S[i,0]/1e6, V[i,0]/1e6, dist[i]*1e3))


diag("solid.lp_sparcap_center_thickness_015.coords", "cap-centre", False)
diag("solid.circumferential_015.coords", "circumferential", True)
