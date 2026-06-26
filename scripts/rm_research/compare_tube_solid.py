"""
Overlay the JAX shell dehom (MSG-TW, material frame) against the FEniCS-solid
dehom on the anisotropic -45 tube, path y2=0, y3>0 (top wall), same FF.

Two drives:
  (a) FORCE  -- each model uses its own 6x6 to get the beam strain (what the
      user asked: "realistic load"); differences include the stiffness gap.
  (b) STRAIN -- both driven by the SAME beam strain (the solid's inv(C6_solid)@FF),
      isolating the stress-RECOVERY difference from the stiffness difference.
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
SOLID = os.path.join(OUT, "tube_solid_toppath.txt")
FF = np.array([2.0e4, 0.0, 0.0, 3.0e2, 1.5e2, 0.0])
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
# solid 6x6 (Deff_srt) printed by the WSL run -- used for the strain-driven case
C6_SOLID = None   # filled from a small re-read if needed; force-drive is primary


def main():
    b = solve_tw_from_yaml(YAML, frac=0.5)
    sol = np.loadtxt(SOLID, skiprows=1)               # y2 y3 S11 S22 S33 S23 S13 S12
    sy3, sS = sol[:, 1], sol[:, 2:8]
    keep = sy3 > 0
    sy3, sS = sy3[keep], sS[keep]
    o = np.argsort(sy3); sy3, sS = sy3[o], sS[o]
    t_sol = (R + H/2 - sy3) / H                        # 0=OML, 1=IML

    coords = np.column_stack([np.zeros_like(sy3), sy3])  # JAX at the SAME y3, y2=0
    jS = np.asarray(stress_at_points(b, coords, beam_force_vabs=FF,
                                     frame="material")["stress"])

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Anisotropic -45 tube, dehom on y2=0,y3>0 (top wall, material frame)\n"
                 "JAX MSG-TW shell vs FEniCS-solid, FF=[2e4,0,0,300,150,0]",
                 fontweight="bold")
    for j, c in enumerate(COMP):
        a = ax.flat[j]
        a.plot(t_sol, jS[:, j]/1e6, "r-o", ms=4, label="JAX MSG-TW (shell)")
        a.plot(t_sol, sS[:, j]/1e6, "g--^", ms=4, label="FEniCS-solid")
        a.set_title(f"$\\sigma_{{{c[1:]}}}$"); a.set_xlabel("depth (0=OML,1=IML)")
        a.set_ylabel(f"{c} (MPa)"); a.grid(True, ls=":", alpha=0.6); a.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "tube_jax_vs_solid.png"), dpi=150); plt.close(fig)
    print("wrote", os.path.join(OUT, "tube_jax_vs_solid.png"))

    print("\nTop-path comparison (MPa), JAX shell vs FEniCS-solid:")
    print(f"  {'depth':>6s}" + "".join(f"{c+'_J':>9s}{c+'_S':>9s}" for c in ["S11", "S22", "S12"]))
    for k in range(0, len(sy3), max(1, len(sy3)//10)):
        row = f"  {t_sol[k]:6.2f}"
        for j in (0, 1, 5):
            row += f"{jS[k,j]/1e6:9.3f}{sS[k,j]/1e6:9.3f}"
        print(row)
    # means and gradients of S11 (axial vs bending split)
    print(f"\n  S11 mean  JAX {jS[:,0].mean()/1e6:.3f}  solid {sS[:,0].mean()/1e6:.3f} MPa")
    print(f"  S11 OML->IML  JAX {(jS[-1,0]-jS[0,0])/1e6:+.3f}  "
          f"solid {(sS[-1,0]-sS[0,0])/1e6:+.3f} MPa  (through-thickness gradient)")


if __name__ == "__main__":
    main()
