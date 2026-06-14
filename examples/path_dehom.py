"""
Through-thickness dehomogenization at the LP spar cap, station 15:
  MSG-TW (JAX)  vs  VABS (.SM)  in the LOCAL MATERIAL (ply/fiber) frame.

The spar-cap paths run OML -> IML, so they show the full ply-by-ply stress
profile through the laminate (the circumferential path only sampled the OML
free surface, where the transverse components are ~0).

Outputs (OML frac=0 and IML frac=1, separate figures per path):
  outputs/st15_leftspar_{OML,IML}.png   (LP sparcap left edge)
  outputs/st15_midspar_{OML,IML}.png    (LP sparcap centre)
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
PDIR = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data"
        r"\opensg-FEniCS\data\st15_path_coords-20260614T203452Z-3-001\st15_path_coords")
SM = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data"
      r"\opensg-FEniCS\data\bar_urc-15-t-0.in.SM")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def load_sm(path):
    d = np.loadtxt(path)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]   # -> S11,S22,S33,S23,S13,S12


def panel(fname, title, z, jax_q2, jax_q3, vabs):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for j, c in enumerate(COMP):
        ax = axes.flat[j]
        ax.plot(z * 1e3, jax_q2[:, j] / 1e6, "r--o", ms=4, label="MSG-TW quad (order 2)")
        ax.plot(z * 1e3, jax_q3[:, j] / 1e6, "b-.s", ms=4, label="MSG-TW cubic (order 3)")
        ax.plot(z * 1e3, vabs[:, j] / 1e6, "g-^", ms=6, label="VABS")
        ax.set_title(f"$\\sigma_{{{c[1:]}}}$  ({c})", fontweight="bold")
        ax.set_xlabel("through-thickness  (mm, OML->IML)")
        ax.set_ylabel(f"{c}  (MPa)"); ax.grid(True, ls=":", alpha=0.7); ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(fname, dpi=150); plt.close(fig)
    print("wrote", fname)


def run(path_file, name, b_oml, b_iml, sm_xy, sm_s):
    coords = np.loadtxt(os.path.join(PDIR, path_file))[:, :2]
    z = np.r_[0.0, np.cumsum(np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])))]
    vabs = sm_s[cKDTree(sm_xy).query(coords)[1]]
    print(f"\n{name}: {len(coords)} through-thickness pts, span {z[-1]*1e3:.1f} mm")
    for tag, b in [("OML", b_oml), ("IML", b_iml)]:
        q2 = stress_at_points(b, coords, beam_force_vabs=FF, frame="material", elem_order=2)["stress"]
        q3 = stress_at_points(b, coords, beam_force_vabs=FF, frame="material", elem_order=3)["stress"]
        d = np.max(np.abs(q2 - q3)) / max(np.max(np.abs(q2)), 1e-30)
        print(f"  {tag}: max|cubic-quad|/scale = {d:.2e}")
        panel(os.path.join(OUT, f"st15_{name}_{tag}.png"),
              f"Station 15 {name} through-thickness (material frame) — {tag}  |  "
              "quad vs cubic step-2 dehom",
              z, q2, q3, vabs)


def main():
    b_oml = solve_tw_from_yaml(SHELL15, frac=0.0)
    b_iml = solve_tw_from_yaml(SHELL15, frac=1.0)
    sm_xy, sm_s = load_sm(SM)
    run("solid.lp_sparcap_left_edge_thickness_015.coords", "leftspar",
        b_oml, b_iml, sm_xy, sm_s)
    run("solid.lp_sparcap_center_thickness_015.coords", "midspar",
        b_oml, b_iml, sm_xy, sm_s)


if __name__ == "__main__":
    main()
