"""
Out-of-plane stress under a SHEAR load (F2=2e4, which gives max transverse-shear
flow at the top of the tube): solid vs JAX shell (Kirchhoff).
Answers: does sigma33/sigma13/sigma23 stay zero for Kirchhoff and RM?
  - sigma33 ~ 0 for both shell models (plane-stress) and for the solid (free faces).
  - transverse shear sigma13/sigma23: Kirchhoff (C1, no shear DOF) is IDENTICALLY 0;
    the solid shows the parabolic through-thickness profile; RM recovers that profile.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import solve_tw_from_yaml, stress_at_points

R, H = 0.0715, 0.008682
OUT = os.path.join(HERE, "..", "outputs", "tube_dehom")
YAML = os.path.join(OUT, "aniso_tube.yaml")
FF = np.array([5.0e3, 2.0e4, 0.0, 0.0, 0.0, 0.0])     # axial + horizontal shear F2
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]


def main():
    b = solve_tw_from_yaml(YAML, frac=0.5)
    sol = np.loadtxt(os.path.join(OUT, "solid_shear_F2_toppath.txt"), skiprows=1)
    sy3, sS = sol[:, 1], sol[:, 2:8]
    o = np.argsort(sy3); sy3, sS = sy3[o], sS[o]
    t = (R + H/2 - sy3)/H
    coords = np.column_stack([np.zeros_like(sy3), sy3])
    jS = np.asarray(stress_at_points(b, coords, beam_force_vabs=FF,
                                     frame="material")["stress"])

    print("Out-of-plane under F2=2e4 shear (top path), MPa:")
    print(f"  {'comp':5s}{'Kirchhoff max|.|':>18s}{'solid max|.|':>14s}")
    for j in (2, 4, 3):   # S33, S13, S23
        print(f"  {COMP[j]:5s}{np.max(np.abs(jS[:,j]))/1e6:18.4f}{np.max(np.abs(sS[:,j]))/1e6:14.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Out-of-plane under F2 shear (top path): Kirchhoff vs solid\n"
                 "(RM recovers the parabolic transverse shear; Kirchhoff is structurally 0)",
                 fontweight="bold")
    for c, j in enumerate((2, 4, 3)):    # S33, S13, S23
        a = ax[c]
        a.plot(t, jS[:, j]/1e6, "b:s", ms=4, label="JAX shell (Kirchhoff)")
        a.plot(t, sS[:, j]/1e6, "g--^", ms=4, label="FEniCS-solid")
        a.set_title(f"$\\sigma_{{{COMP[j][1:]}}}$"); a.set_xlabel("depth (0=OML,1=IML)")
        a.set_ylabel(f"{COMP[j]} (MPa)"); a.grid(True, ls=":", alpha=0.6); a.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(OUT, "tube_oop_shear.png"), dpi=150); plt.close(fig)
    print("wrote", os.path.join(OUT, "tube_oop_shear.png"))


if __name__ == "__main__":
    main()
