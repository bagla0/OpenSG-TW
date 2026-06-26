"""
Station-13 dehom: shell (1Dshell_13, OML) vs FEniCS-solid (2Dsolid_12 = st13)
on a clean through-thickness path (inward normal at a thin upper-surface panel),
same load. Material frame.
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, solve_tw_from_yaml, stress_at_points
from fe_jax.msg_mesh import read_mesh

SHELL = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_13.yaml"
SOLIDF = (r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
          r"\outputs\st12\st13_solid_dehom_full.txt")
OUT = os.path.join(HERE, "..", "outputs", "st12")
FF = np.array([2.0e5, 0.0, 0.0, 3.0e5, 8.0e5, 5.0e5])
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]
TARGET = np.array([1.0, 0.41])      # upper-surface fore panel


def main():
    b = solve_tw_from_yaml(SHELL, frac=0.0)
    n3d, elements, mat_db, layup_db, e2l = load_yaml(SHELL)
    n3d = n3d.copy(); n3d[:, 2] = 0.0
    nodes, cells, lpe = read_mesh(n3d, elements, e2l); xy = nodes[:, :2]
    cen = xy.mean(0)
    mids = np.array([0.5*(xy[c[0]]+xy[c[-1]]) for c in cells])
    e = int(np.argmin(np.hypot(*(mids - TARGET).T)))
    A, B = xy[cells[e][0]], xy[cells[e][-1]]; tv = (B-A)/np.hypot(*(B-A))
    nrm = np.array([tv[1], -tv[0]])
    if (cen - mids[e]) @ nrm < 0: nrm = -nrm                 # inward
    h = float(sum(layup_db[lpe[e]]["thick"]))
    mid = mids[e]
    s = np.linspace(0, h, 17)
    path = mid[None, :] + s[:, None]*nrm[None, :]            # OML -> IML
    print(f"panel '{lpe[e]}' at y2~{mid[0]:.2f},y3~{mid[1]:.2f}, h={h*1e3:.1f} mm")

    jS = np.asarray(stress_at_points(b, path, beam_force_vabs=FF,
                                     frame="material")["stress"])
    sol = np.loadtxt(SOLIDF, skiprows=1)
    _, idx = cKDTree(sol[:, :2]).query(path)
    sS = sol[idx, 2:8]
    t = s/h                                                   # 0=OML,1=IML

    print(f"  {'comp':5s}{'shell max':>12s}{'solid max':>12s}{'% peak':>9s}")
    for j in range(6):
        sm, ss = np.max(np.abs(jS[:, j])), np.max(np.abs(sS[:, j]))
        e2 = 100*(sm-ss)/ss if ss > 5e4 else float('nan')
        print(f"  {COMP[j]:5s}{sm/1e6:12.3f}{ss/1e6:12.3f}{e2:9.1f}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Station 13 dehom, panel at y2~1.0 (h={h*1e3:.0f} mm, material frame): "
                 "shell 1Dshell_13 vs FEniCS-solid", fontweight="bold")
    for j, c in enumerate(COMP):
        a = ax.flat[j]
        a.plot(t, jS[:, j]/1e6, "r-o", ms=4, label="MSG-TW shell")
        a.plot(t, sS[:, j]/1e6, "g--^", ms=4, label="FEniCS-solid")
        a.set_title(f"$\\sigma_{{{c[1:]}}}$"); a.set_xlabel("depth (0=OML,1=IML)")
        a.set_ylabel(f"{c} (MPa)"); a.grid(True, ls=":", alpha=0.6); a.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "st13_dehom_compare.png"), dpi=150); plt.close(fig)
    print("wrote", os.path.join(OUT, "st13_dehom_compare.png"))


if __name__ == "__main__":
    main()
