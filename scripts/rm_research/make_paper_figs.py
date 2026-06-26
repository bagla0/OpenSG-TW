"""
Paper assets for the anisotropic -45 tube at center reference (frac=0.5):
  (1) full 6x6 homogenization comparison: TW-RM, TW-Kirchhoff, FEniCS-solid, Table 3.2;
  (2) mesh + orientation figure (shell 1D circumferential mesh, solid annulus quad mesh).
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import load_yaml, compute_ABD_matrix, timoshenko_from_yaml, solve_tw_from_yaml
from fe_jax.msg_mesh import read_mesh, offset_oml_to_iml, element_e3_from_yaml
from fe_jax.msg_materials import shift_abd_reference
from msg_rm_timo import timoshenko_rm
from transverse_shear import transverse_shear_stiffness

R, H, ANG = 0.0715, 0.008682, -45.0
OUT = os.path.join(HERE, "..", "outputs", "tube_dehom")
YAML = os.path.join(OUT, "aniso_tube.yaml")
FRAC = 0.5
# Table 3.2 classical (EB) values, x1e6
TAB = {"EA": 47.785e6, "ext-tw": -0.93755e6, "GJ": 0.14896e6, "EI": 0.10710e6}


def homo_table():
    solid = np.loadtxt(os.path.join(OUT, "solid_C6_h0.008682.txt"))
    b = solve_tw_from_yaml(YAML, frac=FRAC)
    KF = np.asarray(b["Timo"]); EBkf = np.asarray(b["EB"])
    # RM at center reference
    n3d, elements, mat_db, layup_db, e2l = load_yaml(YAML)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    e3 = element_e3_from_yaml(YAML)
    nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=FRAC)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = -1.0/R * np.ones(len(elems))
    D_by = {ln: shift_abd_reference(
        np.asarray(compute_ABD_matrix(i["thick"], i["angles"], i["mat_names"], mat_db)[0]),
        FRAC*float(sum(i["thick"]))) for ln, i in layup_db.items()}
    G_by = {ln: transverse_shear_stiffness(i["thick"], i["angles"], i["mat_names"], mat_db)[0]
            for ln, i in layup_db.items()}
    RM, _ = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22, p=1)

    terms = [("EA  (C11)", (0, 0), "EA"), ("ext-twist (C14)", (0, 3), "ext-tw"),
             ("GA2 (C22)", (1, 1), None), ("GA3 (C33)", (2, 2), None),
             ("shear-bend (C25)", (1, 4), None), ("GJ  (C44)", (3, 3), "GJ"),
             ("EI2 (C55)", (4, 4), "EI"), ("EI3 (C66)", (5, 5), "EI")]
    lines = []
    hdr = f"{'term':18s}{'TW-RM':>13s}{'TW-Kirch':>13s}{'solid':>13s}{'Table3.2':>13s}"
    print(hdr); lines.append(hdr)
    for nm, (i, j), tk in terms:
        rm, kf, so = RM[i, j], KF[i, j], solid[i, j]
        tv = TAB[tk] if tk else float("nan")
        row = f"{nm:18s}{rm:13.4e}{kf:13.4e}{so:13.4e}" + (
            f"{tv:13.4e}" if tk else f"{'-':>13s}")
        print(row); lines.append(row)
    # % error vs solid (the reference)
    print("\n% error vs FEniCS-solid:")
    lines.append("\n% error vs FEniCS-solid (Timoshenko):")
    for nm, (i, j), tk in terms:
        so = solid[i, j]
        if abs(so) < 1e3:  # skip ~0 terms
            continue
        r = f"  {nm:18s} RM {100*(RM[i,j]-so)/so:+7.2f}%   Kirch {100*(KF[i,j]-so)/so:+7.2f}%"
        print(r); lines.append(r)
    print(f"\n(EB bend2 shell = {EBkf[2,2]:.4e}; Table 3.2 EB EI = {TAB['EI']:.4e})")
    with open(os.path.join(OUT, "homo_table.txt"), "w") as f:
        f.write("\n".join(lines))


def mesh_fig():
    # shell 1D mesh (coarse) + solid annulus quad mesh (coarse) + orientation
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    # (a) shell: contour at mid-surface R, NC elements, e2/e3 arrows
    nc = 48
    th = np.linspace(0, 2*np.pi, nc, endpoint=False)
    xs, ys = R*np.cos(th), R*np.sin(th)
    ax[0].plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], "-o", color="tab:blue", ms=4, lw=1)
    sc = 0.13*R
    for k in range(0, nc, 3):
        t = th[k] + np.pi/nc
        mx, my = R*np.cos(t), R*np.sin(t)
        tang = np.array([-np.sin(t), np.cos(t)])      # e2
        nrm = np.array([-np.cos(t), -np.sin(t)])       # e3 inward (stacking)
        ax[0].arrow(mx, my, sc*tang[0], sc*tang[1], head_width=sc*0.18, color="tab:green")
        ax[0].arrow(mx, my, sc*nrm[0], sc*nrm[1], head_width=sc*0.18, color="tab:red")
    ax[0].plot([], [], color="tab:green", label="$e_2$ (tangent)")
    ax[0].plot([], [], color="tab:red", label="$e_3$ (inward, stacking)")
    ax[0].set_title("(a) Shell model: 1D circumferential mesh + material frame")
    ax[0].set_aspect("equal"); ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_xlabel("$y_2$ (m)"); ax[0].set_ylabel("$y_3$ (m)")
    # (b) solid annulus quad mesh (coarse)
    ncc, nrr = 60, 4
    ri, ro = R - H/2, R + H/2
    rad = np.linspace(ri, ro, nrr+1); ang = np.linspace(0, 2*np.pi, ncc+1)
    for r in rad:
        ax[1].plot(r*np.cos(ang), r*np.sin(ang), color="0.4", lw=0.7)
    for a in ang:
        ax[1].plot([ri*np.cos(a), ro*np.cos(a)], [ri*np.sin(a), ro*np.sin(a)],
                   color="0.4", lw=0.7)
    ax[1].set_title(f"(b) Solid model: annulus quad mesh (h/R={H/R:.3f})")
    ax[1].set_aspect("equal"); ax[1].set_xlabel("$y_2$ (m)"); ax[1].set_ylabel("$y_3$ (m)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "tube_mesh.png"), dpi=150)
    plt.close(fig); print("wrote", os.path.join(OUT, "tube_mesh.png"))


if __name__ == "__main__":
    homo_table()
    mesh_fig()
