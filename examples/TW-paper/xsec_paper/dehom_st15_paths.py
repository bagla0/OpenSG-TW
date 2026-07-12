"""dehom_st15_paths.py -- BUG CHECK: run the SAME RM shell dehom (solve_tw_from_yaml
frac=0 + stress_at_points, material frame, same FF) on THREE paths and report the
in-plane agreement with VABS .SM, to separate a real dehom bug from the known left-edge
cap/web-corner topology limit.
  circumferential  -> should match VABS (path lies on the shell surface)
  cap-centre        -> should match VABS (through-thickness aligned with shell normal)
  left-edge         -> known ~10x off (path folds at the cap/web corner)
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml, stress_at_points

DEH = os.path.expanduser("~/claude_tmp/dehom_st15")
SHELL15 = os.path.expanduser("~/OpenSG-TW-claude/tests/data/1Dshell_15.yaml")
SM = os.path.join(DEH, "bar_urc-15-t-0.in.SM")
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])
PATHS = [("circumferential", "solid.circumferential_015.coords"),
         ("cap-centre", "solid.lp_sparcap_center_thickness_015.coords"),
         ("left-edge", "solid.lp_sparcap_left_edge_thickness_015.coords")]


def load_sm(path):
    d = np.loadtxt(path)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]


sm_xy, sm_s = load_sm(SM)
tree = cKDTree(sm_xy)
bundle = solve_tw_from_yaml(SHELL15, frac=0.0)
print("path              npts   ||S_shell-S_vabs|| / ||S_vabs||   in-plane(S11,S22,S12)   oop(S33,S13,S23)")
for name, fn in PATHS:
    p = os.path.join(DEH, fn)
    if not os.path.exists(p):
        print("%-16s MISSING %s" % (name, fn)); continue
    coords = np.loadtxt(p)[:, :2]
    S = np.asarray(stress_at_points(bundle, coords, beam_force_vabs=FF, frame="material")["stress"])
    V = sm_s[tree.query(coords)[1]]
    rel = np.linalg.norm(S - V) / np.linalg.norm(V) * 100
    ip = np.linalg.norm(S[:, [0, 1, 5]] - V[:, [0, 1, 5]]) / (np.linalg.norm(V[:, [0, 1, 5]]) + 1e-30) * 100
    oop = np.linalg.norm(S[:, [2, 4, 3]] - V[:, [2, 4, 3]]) / (np.linalg.norm(V[:, [2, 4, 3]]) + 1e-30) * 100
    print("%-16s %4d    %6.1f%%                        %6.1f%%                 %6.1f%%"
          % (name, len(coords), rel, ip, oop))
    # peak S11 both, to expose the left-edge saturation
    print("        peak |S11|  shell=%.1f MPa  VABS=%.1f MPa"
          % (np.abs(S[:, 0]).max() / 1e6, np.abs(V[:, 0]).max() / 1e6))
