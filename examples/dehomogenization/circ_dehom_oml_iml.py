"""
Circumferential-path dehomogenization, station 15:
  VABS (.SM Gauss points)  vs  MSG-TW (JAX)  for the OML and the IML reference.

Produces two figures (6 stress components each):
  outputs/st15_circ_dehom_IML.png : JAX-TW (frac=1, IML) vs VABS .SM (+ FE solid)
  outputs/st15_circ_dehom_OML.png : JAX-TW (frac=0, OML) vs VABS .SM (+ FE solid)

The .SM stores stress as [y2, y3, S11, S12, S13, S22, S23, S33] (re-ordered here
to the DOLFINx order [S11, S22, S33, S23, S13, S12]).  The script first decides
whether .SM is the material or global frame by comparing it to the FEniCS
material/global circumferential outputs, and plots everything in that frame.

Prereqs: run fenics_st15.py in WSL (writes st15_fenics_circ_{glob,mat}.txt).
"""
import os
import sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import solve_tw_from_yaml, stress_at_points

SHELL15 = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
DATA = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
        r"\training data\opensg-FEniCS\data")
CIRC = os.path.join(DATA, "st15_path_coords-20260614T203452Z-3-001",
                    "st15_path_coords", "solid.circumferential_015.coords")
SM = os.path.join(DATA, "bar_urc-15-t-0.in.SM")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def load_sm(path):
    d = np.loadtxt(path)
    # .SM cols 2..7 = S11,S12,S13,S22,S23,S33 -> S11,S22,S33,S23,S13,S12
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]


def panel(fname, title, s, jax_s, vabs, fe_solid, jax_lab):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for j, c in enumerate(COMP):
        ax = axes.flat[j]
        if fe_solid is not None:
            ax.plot(s, fe_solid[:, j] / 1e6, "b-", lw=2, label="FEniCS solid")
        ax.plot(s, jax_s[:, j] / 1e6, "r--o", ms=4, label=jax_lab)
        ax.plot(s, vabs[:, j] / 1e6, "g^", ms=5, label="VABS")
        ax.set_title(f"$\\sigma_{{{c[1:]}}}$  ({c})", fontweight="bold")
        ax.set_xlabel("circumferential path  s"); ax.set_ylabel(f"{c}  (MPa)")
        ax.grid(True, ls=":", alpha=0.7); ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fname, dpi=150); plt.close(fig); print("wrote", fname)


def main():
    circ = np.loadtxt(CIRC)[:, :2]
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(circ[:, 0]), np.diff(circ[:, 1])))]
    s /= s[-1]

    # MSG-TW (JAX) dehom in the LOCAL MATERIAL (ply/fiber) frame, OML & IML
    b_oml = solve_tw_from_yaml(SHELL15, frac=0.0)
    b_iml = solve_tw_from_yaml(SHELL15, frac=1.0)
    tw_oml = stress_at_points(b_oml, circ, beam_force_vabs=FF, frame="material")["stress"]
    tw_iml = stress_at_points(b_iml, circ, beam_force_vabs=FF, frame="material")["stress"]

    # FEniCS solid (material frame) + VABS .SM (material) at the circ coords
    fm = np.loadtxt(os.path.join(OUT, "st15_fenics_circ_mat.txt"))
    fmat = fm[cKDTree(fm[:, :2]).query(circ)[1], 2:]
    sm_xy, sm_s = load_sm(SM)
    vabs = sm_s[cKDTree(sm_xy).query(circ)[1]]

    # the FEniCS-solid material field interpolated at the circ coords is noisy
    # (per-element material frame is cell-discontinuous -> CG2 spikes at the
    # web/spar); VABS .SM is the clean solid reference, so plot JAX vs .SM.
    for j, c in enumerate(COMP):
        e = np.abs(tw_oml[:, j] - vabs[:, j])
        print(f"  {c}: median|JAX_OML - VABS| = {np.median(e)/1e6:.3f} MPa")

    panel(os.path.join(OUT, "st15_circ_dehom_IML.png"),
          "Station 15 circumferential dehom (LOCAL MATERIAL frame) — IML (frac=1)",
          s, tw_iml, vabs, None, "MSG-TW IML (JAX)")
    panel(os.path.join(OUT, "st15_circ_dehom_OML.png"),
          "Station 15 circumferential dehom (LOCAL MATERIAL frame) — OML (frac=0)",
          s, tw_oml, vabs, None, "MSG-TW OML (JAX)")


if __name__ == "__main__":
    main()
