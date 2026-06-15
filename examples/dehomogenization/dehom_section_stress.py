"""
MSG-TW dehomogenization -> 3D stress field over a cross-section, in the local
MATERIAL frame, written in a VABS .SM-like text format, plus a stress comparison
against the VABS .SM along a user-supplied .coords path.

Given a beam force FF, the two-step MSG-TW dehom recovers the 3D stress at any
cross-section point (project to the 1D reference -> through-thickness plate
model).  This script:

  1. Samples the WHOLE cross-section (every shell element x arc gauss points x
     through-thickness points) and writes outputs/dehomo/<stem>_section_stress.txt
     with rows  [y2  y3  S11 S22 S33 S23 S13 S12]  (local material frame) -- the
     JAX-TW analogue of the VABS .SM full-field stress dump.

  2. Reads a .coords path (circumferential here), recovers the stress along it,
     matches the VABS .SM (nearest gauss point), and writes
     outputs/dehomo/<coords_stem>_stress_vs_vabs.png : 6 panels, x = y2
     normalized so the first path coord is 0 and the last is 1, y = stress.

Usage
-----
    python dehom_section_stress.py [<cross_section.yaml>] [<path.coords>]
"""
import os
import sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import solve_tw_from_yaml, stress_at_points

DEFAULT_YAML = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
DEFAULT_COORDS = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data"
                  r"\opensg-FEniCS\data\st15_path_coords-20260614T203452Z-3-001"
                  r"\st15_path_coords\solid.circumferential_015.coords")
SM = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data"
      r"\opensg-FEniCS\data\bar_urc-15-t-0.in.SM")
OUT_DIR = os.path.join(HERE, "..", "..", "outputs", "dehomo")
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]            # output (Voigt) order
# station-15 beam force (VABS order [F1,F2,F3,M1,M2,M3])
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def load_sm(path):
    """VABS .SM -> (xy, stress) with stress reordered to [S11,S22,S33,S23,S13,S12]."""
    d = np.loadtxt(path)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]       # .SM cols S11,S12,S13,S22,S23,S33


def section_points(bundle, n_thick=5):
    """Every element's arc gauss points swept through the thickness (OML->IML).

    Returns the (P,2) cross-section coordinates -- the JAX-TW 'gauss points'."""
    corners = np.asarray(bundle["corners"]); rc = np.asarray(bundle["red_cells"])
    xd2 = np.asarray(bundle["xd2"]); xd3 = np.asarray(bundle["xd3"])
    xi_q = np.asarray(bundle["xi_q"]); cen = corners.mean(0)
    layups = bundle["layup_per_elem"]; ldb = bundle["layup_db"]
    pts = []
    for e in range(rc.shape[0]):
        n0 = corners[rc[e, 0]]; n1 = corners[rc[e, 1]]
        nin = np.array([xd3[e], -xd2[e]]); mid = 0.5 * (n0 + n1)
        if (cen - mid) @ nin < 0.0:
            nin = -nin
        h = float(sum(ldb[layups[e]]["thick"]))
        for xi in xi_q:
            pa = (1.0 - xi) * n0 + xi * n1
            for z in np.linspace(0.0, h, n_thick):
                pts.append(pa + z * nin)
    return np.array(pts)


def write_section_txt(bundle, stem):
    pts = section_points(bundle)
    S = stress_at_points(bundle, pts, beam_force_vabs=FF, frame="material")["stress"]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{stem}_section_stress.txt")
    header = ("MSG-TW dehomogenization 3D stress over the cross-section "
              "(local material frame)\n"
              "columns: y2  y3  S11  S22  S33  S23  S13  S12")
    np.savetxt(path, np.column_stack([pts, S]), fmt="% .8e", header=header)
    print(f"wrote {path}   ({len(pts)} points)")
    return path


def plot_path_vs_vabs(bundle, coords_path):
    coords = np.loadtxt(coords_path)[:, :2]
    S = stress_at_points(bundle, coords, beam_force_vabs=FF, frame="material")["stress"]
    sm_xy, sm_s = load_sm(SM)
    vabs = sm_s[cKDTree(sm_xy).query(coords)[1]]

    y2 = coords[:, 0]
    denom = y2[-1] - y2[0]
    xnd = (y2 - y2[0]) / (denom if abs(denom) > 1e-12 else 1.0)   # first=0, last=1

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle("MSG-TW dehom vs VABS (.SM) along path — local material frame",
                 fontsize=13, fontweight="bold")
    for j, c in enumerate(COMP):
        ax = axes.flat[j]
        ax.plot(xnd, S[:, j] / 1e6, "r-o", ms=3.5, label="MSG-TW")
        ax.plot(xnd, vabs[:, j] / 1e6, "g--^", ms=4, label="VABS (.SM)")
        ax.set_title(f"$\\sigma_{{{c[1:]}}}$  ({c})", fontweight="bold")
        ax.set_xlabel("normalized y2  (first coord = 0, last = 1)")
        ax.set_ylabel(f"{c}  (MPa)"); ax.grid(True, ls=":", alpha=0.6); ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(coords_path))[0]
    png = os.path.join(OUT_DIR, f"{stem}_stress_vs_vabs.png")
    fig.savefig(png, dpi=150); plt.close(fig)
    print(f"wrote {png}")


def main():
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_YAML
    coords_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_COORDS
    bundle = solve_tw_from_yaml(yaml_path, frac=0.0)           # OML reference
    stem = os.path.splitext(os.path.basename(yaml_path))[0]
    write_section_txt(bundle, stem)
    plot_path_vs_vabs(bundle, coords_path)


if __name__ == "__main__":
    main()
