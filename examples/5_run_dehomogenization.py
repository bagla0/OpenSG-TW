"""
Two-step MSG-TW dehomogenization driver.

Solves the thin-walled cross-section (Hermite C1 Timoshenko), then recovers the
local strain field for an applied beam force in two steps:

  Step 1  shell strains  : 6 plate strains [eps11, eps22, gamma12, kappa11,
                           kappa22, kappa12] at every cross-section arc point,
                           from the EB warping V0 and shear warping V1.
  Step 2  plate dehom    : per line element, the 6 shell strains drive the MSG
                           plate (through-thickness 1D SG) model to recover the
                           pointwise 3D strain/stress across the thickness.

Usage
-----
    python examples/run_dehomogenization.py [path/to/cross_section.yaml]

The beam force is in VABS order [F1, F2, F3, M1, M2, M3] = [axial, shear-2,
shear-3, torsion, bending-2, bending-3]; edit ``FF`` below for your load case.
"""
import sys
import os
import numpy as np
import jax

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONIOENCODING"] = "utf-8"
jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=4, linewidth=120)


def _fmt(v):
    return "[" + " ".join(f"{x: .4e}" for x in np.asarray(v)) + "]"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
from fe_jax import solve_tw_from_yaml, dehomogenize


def run(yaml_path, FF):
    print("=" * 64)
    print("MSG-TW Dehomogenization  |  shell strains -> 3D through-thickness")
    print("=" * 64)
    print(f"YAML       : {yaml_path}")
    print(f"Beam force : {FF}  (VABS [F1,F2,F3,M1,M2,M3])\n")

    bundle = solve_tw_from_yaml(yaml_path)
    out = dehomogenize(yaml_path, FF, n_eval_per_elem=5, bundle=bundle)

    st = out["macro"]
    print("Beam macro strain st [eps11, gamma12, gamma13, kappa1, kappa2, kappa3]:")
    print(f"  {_fmt(st)}\n")

    ss = out["shell_strain_elem"]
    n_elem = ss.shape[0]
    print(f"Step 1 — shell strains per element ({n_elem} elements)")
    print("  [eps11, eps22, gamma12, kappa11, kappa22, kappa12]   (max |.| per col)")
    print(f"  max : {_fmt(np.max(np.abs(ss), axis=0))}")
    # most-strained element by max |eps11| at the surface
    e_hot = int(np.argmax(np.abs(ss[:, 0])))
    print(f"  e={e_hot:<3d}: {_fmt(ss[e_hot])}\n")

    print(f"Step 2 — 3D strain/stress across thickness (element {e_hot}, "
          f"layup '{out['elem'][e_hot]['layup']}')")
    ed = out["elem"][e_hot]
    z, Gam, Sig = ed["z"], ed["strain_3d"], ed["stress_3d"]
    print("   z         e11        e22        e33        g23        g13        g12")
    idx = np.linspace(0, len(z) - 1, min(7, len(z))).astype(int)
    for i in idx:
        print(f"  {z[i]:8.5f}  " + " ".join(f"{Gam[i, j]:10.3e}" for j in range(6)))
    print("\n   z       s11(max-stress component shown across thickness)")
    print(f"  von-Mises-ish max |sigma| = {np.max(np.abs(Sig)):.4e}")
    print("\nDone.")
    return out


if __name__ == "__main__":
    default_yaml = os.path.join(
        r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG", "1Dshell_0.yaml")
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else default_yaml
    # Example load: axial + bending-2 (edit for your case).
    FF = np.array([1.0e5, 0.0, 0.0, 0.0, 1.0e4, 0.0])
    run(yaml_path, FF)
