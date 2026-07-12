"""dehom_st15_leftspar.py -- EX4 dehomogenization, station-15 (BAR-URC) LP spar-cap
LEFT-EDGE through-thickness path.  Recovers the 3-D stress from the RM SHELL two-step
dehomogenization and compares it, component by component, against the VABS .SM stress
recovery (the benchmark).  The OpenSG-FEniCS solid-dehom curve is added in step 2 once
the solid recovery is wired (see check at the bottom).

  RM shell : opensg_jax.solve_tw_from_yaml + stress_at_points (frame='material')
  VABS     : bar_urc-15-t-0.in.SM gauss points, matched to the path by nearest neighbour
  -> results/dehom_st15_leftspar.npz + figures/dehom_st15_leftspar.png
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml, stress_at_points

DEH = os.path.join(HERE, "..", "..", "..", "examples", "data", "dehom_st15")
SHELL15 = os.path.expanduser("~/OpenSG-TW-claude/tests/data/1Dshell_15.yaml")
SM = os.path.join(DEH, "bar_urc-15-t-0.in.SM")
COORDS = os.path.join(DEH, "solid.lp_sparcap_left_edge_thickness_015.coords")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
# station-15 beam force in VABS order [F1,F2,F3,M1,M2,M3]
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def load_sm(path):
    d = np.loadtxt(path)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]   # -> [S11,S22,S33,S23,S13,S12]


coords = np.loadtxt(COORDS)[:, :2]
z_mm = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))] * 1e3
print("path points:", len(coords), " thickness span %.1f mm" % z_mm[-1])

bundle = solve_tw_from_yaml(SHELL15, frac=0.0)               # OML reference
S_shell = np.asarray(stress_at_points(bundle, coords, beam_force_vabs=FF,
                                      frame="material")["stress"])
sm_xy, sm_s = load_sm(SM)
S_vabs = sm_s[cKDTree(sm_xy).query(coords)[1]]
print("shell/VABS stress recovered, shape", S_shell.shape, S_vabs.shape)

np.savez(os.path.join(OUT, "dehom_st15_leftspar.npz"),
         coords=coords, z_mm=z_mm, S_shell=S_shell, S_vabs=S_vabs, comp=COMP, FF=FF)

fig, ax = plt.subplots(2, 3, figsize=(11, 7))
fig.suptitle("Station-15 LP spar-cap left-edge through-thickness: 3-D stress recovery",
             fontsize=12, fontweight="bold")
for k, c in enumerate(COMP):
    a = ax.flat[k]; oop = c in ("S33", "S13", "S23")
    a.plot(z_mm, S_shell[:, k] / 1e6, "r--o", ms=3.5, label="RM shell (two-step)")
    a.plot(z_mm, S_vabs[:, k] / 1e6, "g-^", ms=4, label="VABS (.SM)")
    a.set_title(r"$\sigma_{%s}$" % c[1:] + ("  [out-of-plane]" if oop else ""),
                color=("darkred" if oop else "black"), fontsize=10)
    a.set_xlabel("through-thickness (mm, OML$\\to$IML)"); a.set_ylabel("%s (MPa)" % c)
    a.grid(True, ls=":", alpha=0.6); a.legend(fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
p = os.path.join(FIG, "dehom_st15_leftspar.png")
fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
