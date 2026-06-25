"""
Strip-WIDTH convergence study (isotropic), fixed thickness h, varying width W.

Computes the MSG-TW shell Timoshenko 6x6 in RM and Kirchhoff (centre reference)
for a range of widths and compares to the FEniCS 2D-solid (VABS) reference
(strip_width_solid.py -> data/strip_width_solid.csv).  As W grows (h/W -> 0) the
strip becomes a slender plate and both shell models converge to the solid; the
study shows where RM is more accurate than Kirchhoff (chiefly torsion GJ for the
stubbier, smaller-W strips).

Outputs (report/):
  strip_width_table.txt -- RM/KF/solid + %error vs W;
  strip_width_err.png   -- % error vs the solid for each non-zero stiffness vs W,
                           RM and KF (convergence as W increases).
This file runs RM and KF (Windows, JAX).  h = 0.05 m, isotropic E=70 GPa, nu=0.3.
"""
import os, sys
import numpy as np
import yaml as _yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rm"))
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix, timoshenko_from_yaml
from fe_jax.msg_mesh import read_mesh, offset_oml_to_iml, element_e3_from_yaml
from fe_jax.msg_materials import shift_abd_reference
from msg_rm_timo import timoshenko_rm
from transverse_shear import transverse_shear_stiffness

plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
                     "figure.titlesize": 19, "lines.linewidth": 2.4, "lines.markersize": 9})

H, N = 0.05, 121
WIDTHS = [0.1, 0.2, 0.35, 0.6, 1.0, 1.6, 2.5]
ISO = {"E": [70e9, 70e9, 70e9], "G": [26.923e9]*3, "nu": [0.3, 0.3, 0.3]}
DATA = os.path.join(HERE, "data"); REP = os.path.join(HERE, "report")
os.makedirs(REP, exist_ok=True)
DIAG = [(0, "C11", "EA"), (1, "C22", "GA2"), (2, "C33", "GA3"),
        (3, "C44", "GJ"), (4, "C55", "EI2"), (5, "C66", "EI3")]


def gen_yaml(path, W):
    y2 = np.linspace(-W/2.0, W/2.0, N)
    data = {"nodes": [[float(y), float(H/2.0), 0.0] for y in y2],
            "elements": [[k+1, k+2] for k in range(N-1)],
            "sets": {"element": [{"name": "strip", "labels": list(range(1, N))}]},
            "sections": [{"elementSet": "strip", "layup": [["mat", float(H), 0.0]]}],
            "materials": [{"name": "mat", "density": 1800.0, "elastic": ISO}],
            "elementOrientations": [[0., 0., 1., 1., 0., 0., 0., -1., 0.] for _ in range(N-1)]}
    with open(path, "w") as f:
        _yaml.safe_dump(data, f)


def shell_6x6(yaml_path, frac):
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    if frac:
        e3 = element_e3_from_yaml(yaml_path)
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]; k22 = np.zeros(len(elems))
    def D_of(i):
        a = np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0])
        return shift_abd_reference(a, frac*float(sum(i["thick"]))) if frac else a
    D_by = {ln: D_of(i) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}
    RM, _ = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1)
    _, KF, _ = timoshenko_from_yaml(yaml_path, frac=frac)
    return np.asarray(RM), np.asarray(KF)


def load_solid():
    sol = {}
    with open(os.path.join(DATA, "strip_width_solid.csv")) as f:
        hdr = f.readline().strip().split(","); ci = hdr.index("C11")
        for line in f:
            p = line.strip().split(",")
            sol[round(float(p[0]), 6)] = np.array([float(x) for x in p[ci:ci+36]]).reshape(6, 6)
    return sol


def main():
    solid = load_solid()
    res = {}
    for W in WIDTHS:
        yml = os.path.join(DATA, f"stripW_{W}.yaml")
        gen_yaml(yml, W)
        rm, kf = shell_6x6(yml, 0.5)
        res[W] = {"RM": rm, "KF": kf, "solid": solid[round(W, 6)]}

    lines = [f"Strip-WIDTH convergence (isotropic, h={H} m fixed). "
             "RM/KF (centre) vs FEniCS-solid.  Order [ext,V2,V3,tw,M2,M3].", ""]
    for idx, cij, eng in DIAG:
        lines.append(f"=== {cij} ({eng}) ===")
        lines.append(f"{'W':>6s}{'h/W':>7s}{'RM':>13s}{'KF':>13s}{'solid':>13s}{'RM%':>8s}{'KF%':>8s}")
        for W in WIDTHS:
            sv = res[W]["solid"][idx, idx]; rm = res[W]["RM"][idx, idx]; kf = res[W]["KF"][idx, idx]
            lines.append(f"{W:6.2f}{H/W:7.3f}{rm:13.4e}{kf:13.4e}{sv:13.4e}"
                         f"{100*(rm-sv)/sv:8.1f}{100*(kf-sv)/sv:8.1f}")
        lines.append("")
    table = "\n".join(lines); print(table)
    with open(os.path.join(REP, "strip_width_table.txt"), "w") as f:
        f.write(table + "\n")

    Ws = np.array(WIDTHS)
    fig, ax = plt.subplots(3, 2, figsize=(13, 15))
    fig.suptitle(f"Isotropic strip, width convergence (h={H} m fixed): "
                 "shell-vs-solid \\% error", fontweight="bold")
    for k, (idx, cij, eng) in enumerate(DIAG):
        a = ax.flat[k]
        rmp = [100*(res[W]["RM"][idx, idx]-res[W]["solid"][idx, idx])/res[W]["solid"][idx, idx] for W in WIDTHS]
        kfp = [100*(res[W]["KF"][idx, idx]-res[W]["solid"][idx, idx])/res[W]["solid"][idx, idx] for W in WIDTHS]
        a.plot(Ws, rmp, "-o", color="#1f77b4", label="RM")
        a.plot(Ws, kfp, "--s", color="#d62728", label="Kirchhoff")
        a.axhline(0, color="0.45", lw=1.0)
        fin = [v for v in rmp+kfp if abs(v) <= 100]
        M = max(3.0, 1.25*max(abs(v) for v in fin)) if fin else 30.0
        a.set_ylim(-M, M)
        a.set_title(f"{cij} ({eng})"); a.set_xlabel("width $W$ [m]   ($h/W=0.5$ at left $\\to$ $0.02$ at right)")
        a.set_ylabel("\\% error vs solid"); a.set_xscale("log")
        a.grid(True, which="both", ls=":", alpha=0.6)
    ax.flat[0].legend(loc="best")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(REP, "strip_width_err.png"), dpi=150); plt.close(fig)
    print("\nwrote report/strip_width_table.txt, strip_width_err.png")


if __name__ == "__main__":
    main()
