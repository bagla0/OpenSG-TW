"""
Strain-driven shell-vs-solid: drive BOTH with the SAME generalized beam strain
(pure eps11=1, then pure kappa2=1), so there is NO stiffness ambiguity -- any
sigma11(z) difference on the top path is the 3D stress-RECOVERY difference
(thin-shell CLT vs 3D solid for the thick unsymmetric -45 ply).
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


def main():
    b = solve_tw_from_yaml(YAML, frac=0.5)
    print("bundle keys:", [k for k in b.keys()])
    EB = np.asarray(b["EB"])
    Ceff = np.asarray(b.get("Ceff", b.get("C6", np.zeros((6, 6)))))
    print(f"shell EB  EI(bend2) = {EB[2,2]:.4e}")
    if Ceff.shape == (6, 6) and Ceff.any():
        print(f"shell Timo EI(bend2,idx4) = {Ceff[4,4]:.4e}   (solid Timo 1.2281e5)")

    cases = [("eps11", np.array([1., 0, 0, 0, 0, 0]), "solid_strain_eps11.txt"),
             ("kappa2", np.array([0., 0, 0, 0, 1., 0]), "solid_strain_kappa2.txt")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Strain-driven sigma11(z) on top path: shell vs solid (same beam strain)",
                 fontweight="bold")
    for c, (nm, st, sfile) in enumerate(cases):
        sol = np.loadtxt(os.path.join(OUT, sfile), skiprows=1)
        sy3, sS11 = sol[:, 1], sol[:, 2]
        o = np.argsort(sy3); sy3, sS11 = sy3[o], sS11[o]
        coords = np.column_stack([np.zeros_like(sy3), sy3])
        jS11 = np.asarray(stress_at_points(b, coords, beam_strain=st,
                                           frame="material")["stress"])[:, 0]
        t = (R + H/2 - sy3)/H
        a = ax[c]
        a.plot(t, jS11/1e6, "r-o", ms=4, label="JAX shell")
        a.plot(t, sS11/1e6, "g--^", ms=4, label="FEniCS-solid")
        a.set_title(f"pure {nm}=1"); a.set_xlabel("depth (0=OML,1=IML)")
        a.set_ylabel("sigma11 (MPa)"); a.grid(True, ls=":", alpha=0.6); a.legend()
        gj = (jS11[-1]-jS11[0])/1e6; gs = (sS11[-1]-sS11[0])/1e6
        print(f"  pure {nm:7s}: shell mean {jS11.mean()/1e6:9.1f} grad {gj:+9.2f} | "
              f"solid mean {sS11.mean()/1e6:9.1f} grad {gs:+9.2f} | "
              f"mean ratio {jS11.mean()/sS11.mean():.3f} grad ratio {gj/gs:.3f}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "tube_strain_driven.png"), dpi=150); plt.close(fig)
    print("wrote", os.path.join(OUT, "tube_strain_driven.png"))


if __name__ == "__main__":
    main()
