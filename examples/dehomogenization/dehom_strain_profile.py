"""
Through-thickness 3D STRAIN from the MSG-TW *two-step* dehomogenization,
in the LOCAL MATERIAL (ply/fiber) frame, at the station-15 spar cap.

This is the strain the TW dehom actually uses -- NOT an arbitrary test strain:

  beam force FF  --(inv Timoshenko)-->  beam strain
                 --(recover_shell_strains)-->  6 shell strains [e11,e22,g12,k11,k22,k12]
                 --(plate 1D-SG warping V0)-->  6 through-thickness 3D strains.

The recovered shell strain at the spar cap is printed first (the step-1 output
that drives step 2), then the full 6-component 3D strain is tabulated and
plotted OML->IML.

Outputs:
  outputs/st15_strain_{leftspar,midspar}_material.png
  + a printed table of eps_11..eps_12 (material frame) through the thickness.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import solve_tw_from_yaml, stress_at_points
from fe_jax.msg_dehom import recover_shell_strains, _project_point

SHELL15 = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
PDIR = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data"
        r"\opensg-FEniCS\data\st15_path_coords-20260614T203452Z-3-001\st15_path_coords")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
# 3D strain order returned by stress_at_points: [e11,e22,e33,2e23,2e13,2e12]
COMP = ["e11", "e22", "e33", "g23", "g13", "g12"]
SHELL = ["e11", "e22", "g12", "k11", "k22", "k12"]
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def panel(fname, title, z, strn):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for j, c in enumerate(COMP):
        ax = axes.flat[j]
        oop = c in ("e33", "g13", "g23")
        ax.plot(strn[:, j] * 1e6, z * 1e3, "b.-", ms=6)
        ax.set_title(f"$\\{('gamma' if c[0]=='g' else 'varepsilon')}_{{{c[1:]}}}$"
                     + ("  [out-of-plane]" if oop else ""),
                     fontweight="bold", color=("darkred" if oop else "black"))
        ax.set_xlabel(f"{c}  (microstrain)")
        ax.set_ylabel("through-thickness  (mm, OML->IML)")
        ax.invert_yaxis(); ax.grid(True, ls=":", alpha=0.7)
        ax.axvline(0, color="0.6", lw=0.8)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(fname, dpi=150); plt.close(fig)
    print("wrote", fname)


def run(bundle, path_file, name):
    coords = np.loadtxt(os.path.join(PDIR, path_file))[:, :2]
    z = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))]

    # step-1 shell strain recovered at this spar-cap arc location (the TW dehom input)
    e, xi, _ = _project_point(
        np.asarray(bundle["corners"]), np.asarray(bundle["red_cells"]), coords[0])
    rec = recover_shell_strains(bundle, beam_force_vabs=FF, xi_eval=[xi])
    sh = rec["shell_strain"][e, 0]
    print(f"\n=== {name}: recovered SHELL strain (step 1, drives the dehom) ===")
    print("  " + "  ".join(f"{n}={v:+.4e}" for n, v in zip(SHELL, sh)))

    out = stress_at_points(bundle, coords, beam_force_vabs=FF, frame="material")
    strn = out["strain"]
    print(f"  through-thickness 3D STRAIN (material frame), {len(z)} pts OML->IML:")
    print("    z(mm)  " + "  ".join(f"{c:>10}" for c in COMP) + "   (microstrain)")
    step = max(1, len(z) // 12)
    for k in range(0, len(z), step):
        print(f"   {z[k]*1e3:6.2f} " +
              "  ".join(f"{strn[k,j]*1e6:10.2f}" for j in range(6)))
    panel(os.path.join(OUT, f"st15_strain_{name}_material.png"),
          f"Station 15 {name} — TW-recovered 3D strain (material frame, OML->IML)",
          z, strn)


def main():
    bundle = solve_tw_from_yaml(SHELL15, frac=0.0)
    run(bundle, "solid.lp_sparcap_left_edge_thickness_015.coords", "leftspar")
    run(bundle, "solid.lp_sparcap_center_thickness_015.coords", "midspar")


if __name__ == "__main__":
    main()
