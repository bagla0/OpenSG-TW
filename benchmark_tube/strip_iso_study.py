"""
Detailed ISOTROPIC plate-strip study (h/W sweep, thin -> thick).

Computes the thin-walled MSG-shell Timoshenko 6x6 in BOTH wall kinematics --
Reissner-Mindlin (RM, C0) and Kirchhoff (KF, C1) -- at the OML and centre
references, and compares every non-zero stiffness against:
  * the FEniCS 2D-solid cross-sectional analysis (VABS-equivalent), and
  * the MSG-TW analytical closed form (OpenSG / Deo-Yu thin-walled theory).

Strip: width W=1 m, thickness h, isotropic E=70 GPa, nu=0.3 (G=26.923 GPa).
Timoshenko order [ext, shear2, shear3, twist, bend2, bend3] =
  C11=EA, C22=GA2, C33=GA3, C44=GJ, C55=EI2, C66=EI3.

Outputs (written to report/):
  * strip_iso_table.txt  -- term-by-term RM/KF/solid/analytic table (+ % vs solid),
    OML and centre, every h/W;
  * strip_iso_abs.png    -- each non-zero stiffness vs h/W (log-log), all sources;
  * strip_iso_err.png    -- % error vs the solid for each non-zero stiffness,
    RM/KF at OML and centre, plus the analytic.

This file runs RM and KF (Windows, JAX).  The FEniCS-solid reference is precomputed
in WSL by strip_iso_solid.py -> data/strip_iso_solid.csv (re-run it if you change
the h/W list below).

  RM solve   : msg_rm_timo.timoshenko_rm  (C0 Lagrange, transverse shear retained,
               selective reduced integration; MSG plate G block, no shear factor)
  KF solve   : fe_jax.timoshenko_from_yaml (C1 cubic Hermite, no transverse shear)
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml as _yaml

HERE = os.path.dirname(__file__)
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

W, N = 1.0, 161
E, NU = 70e9, 0.3
G = E/(2*(1+NU))
ISO = {"E": [E, E, E], "G": [G, G, G], "nu": [NU, NU, NU]}
HW = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
DATA = os.path.join(HERE, "data")
REP = os.path.join(HERE, "report"); os.makedirs(REP, exist_ok=True)
DIAG = [(0, "C11", "EA"), (1, "C22", "GA2"), (2, "C33", "GA3"),
        (3, "C44", "GJ"), (4, "C55", "EI2"), (5, "C66", "EI3")]


# ---------------------------------------------------------------- analytical
def analytic(hw):
    """MSG-TW isotropic-strip closed form (OpenSG / Deo-Yu): the 6 diagonals."""
    h = hw*W
    bb = h/W
    J = W*h**3*(1.0/3 - 0.21*bb*(1 - bb**4/12))      # exact St-Venant rectangle
    return {"C11": E*W*h,            # EA
            "C22": 5.0/6*G*W*h,      # GA2 = (MSG plate G = 5/6 Gh) x W
            "C33": 5.0/6*G*W*h,      # GA3 (compact 1D; over-predicts the 2D solid)
            "C44": G*J,              # GJ
            "C55": E*W*h**3/12,      # EI2 (weak, ~h^3)
            "C66": E*h*W**3/12}      # EI3 (strong, ~W^3)


# ---------------------------------------------------------------- shell (RM, KF)
def gen_yaml(path, H):
    """Flat strip: nodes along y2 in [-W/2,W/2] at the top OML (y3=H/2); e3=(0,-1)."""
    y2 = np.linspace(-W/2.0, W/2.0, N)
    data = {"nodes": [[float(y), float(H/2.0), 0.0] for y in y2],
            "elements": [[k+1, k+2] for k in range(N-1)],
            "sets": {"element": [{"name": "strip", "labels": list(range(1, N))}]},
            "sections": [{"elementSet": "strip", "layup": [["mat", float(H), 0.0]]}],
            "materials": [{"name": "mat", "density": 1800.0, "elastic": ISO}],
            "elementOrientations": [[0., 0., 1., 1., 0., 0., 0., -1., 0.] for _ in range(N-1)]}
    with open(path, "w") as f:
        _yaml.safe_dump(data, f)


def shell_6x6(yaml_path, H, frac):
    """Return (RM 6x6, KF 6x6) at reference frac (0=OML, 0.5=centre)."""
    n3d, elements, mat_db, layup_db, e2l = load_yaml(yaml_path)
    nodes, cells, lpe = read_mesh(n3d, elements, e2l)
    if frac:
        e3 = element_e3_from_yaml(yaml_path)
        nodes = offset_oml_to_iml(nodes, cells, lpe, layup_db, elem_e3=e3, frac=frac)
    nodes2d = nodes[:, :2]; elems = cells[:, [0, 1]]
    k22 = np.zeros(len(elems))                                  # flat
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
    path = os.path.join(DATA, "strip_iso_solid.csv")
    sol = {}
    with open(path) as f:
        hdr = f.readline().strip().split(","); ci = hdr.index("C11")
        for line in f:
            p = line.strip().split(",")
            sol[float(p[0])] = np.array([float(x) for x in p[ci:ci+36]]).reshape(6, 6)
    return sol


# ---------------------------------------------------------------- main
def main():
    solid = load_solid()
    res = {}                                                    # (hw) -> dict of arrays
    for hw in HW:
        H = hw*W
        yml = os.path.join(DATA, f"stripiso_{hw}.yaml")
        gen_yaml(yml, H)
        rm_o, kf_o = shell_6x6(yml, H, 0.0)                     # OML
        rm_c, kf_c = shell_6x6(yml, H, 0.5)                     # centre
        res[hw] = {"RM_OML": rm_o, "KF_OML": kf_o, "RM_ctr": rm_c, "KF_ctr": kf_c,
                   "analytic": analytic(hw), "solid": solid.get(hw)}

    # ---- table ----
    lines = ["Isotropic plate-strip: Timoshenko stiffness (SI). "
             "RM/KF shell vs FEniCS 2D-solid (VABS) vs MSG-TW analytic.",
             "Order [ext, shear2, shear3, twist, bend2, bend3].", ""]
    for idx, cij, eng in DIAG:
        lines.append(f"=== {cij} ({eng}) ===")
        lines.append(f"{'h/W':>5s}{'RM-ctr':>13s}{'KF-ctr':>13s}{'RM-OML':>13s}"
                     f"{'KF-OML':>13s}{'VABS-solid':>13s}{'analytic':>13s}"
                     f"{'RMc%':>7s}{'KFc%':>7s}{'an%':>7s}")
        for hw in HW:
            r = res[hw]; sv = r["solid"][idx, idx]; an = r["analytic"][cij]
            rc, kc = r["RM_ctr"][idx, idx], r["KF_ctr"][idx, idx]
            ro, ko = r["RM_OML"][idx, idx], r["KF_OML"][idx, idx]
            pe = lambda x: 100*(x-sv)/sv
            lines.append(f"{hw:5.2f}{rc:13.4e}{kc:13.4e}{ro:13.4e}{ko:13.4e}"
                         f"{sv:13.4e}{an:13.4e}{pe(rc):7.1f}{pe(kc):7.1f}{pe(an):7.1f}")
        lines.append("")
    # OML-only reference couplings (parallel-axis artifacts; solid is centroidal)
    lines.append("=== OML-reference couplings (artifacts; centre & solid ~0) ===")
    lines.append(f"{'h/W':>5s}{'C15 ext-b2 (RM/KF)':>26s}{'C24 sh2-tw (RM/KF)':>26s}")
    for hw in HW:
        r = res[hw]
        lines.append(f"{hw:5.2f}   RM {r['RM_OML'][0,4]:.3e}/{r['KF_OML'][0,4]:.3e}"
                     f"   {r['RM_OML'][1,3]:.3e}/{r['KF_OML'][1,3]:.3e}")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(REP, "strip_iso_table.txt"), "w") as f:
        f.write(table + "\n")

    # ---- plot 1: absolute stiffness vs h/W (log-log), all sources ----
    fig, ax = plt.subplots(3, 2, figsize=(13, 15))
    fig.suptitle("Isotropic strip: Timoshenko stiffness vs $h/W$ "
                 "(shell RM/KF vs FEniCS-solid vs MSG-TW analytic)", fontweight="bold")
    hw = np.array(HW)
    for k, (idx, cij, eng) in enumerate(DIAG):
        a = ax.flat[k]
        sv = np.array([res[h]["solid"][idx, idx] for h in HW])
        an = np.array([res[h]["analytic"][cij] for h in HW])
        a.loglog(hw, sv, "k-D", lw=2.6, ms=8, label="FEniCS-solid (VABS)")
        a.loglog(hw, an, ":", color="0.35", marker="P", label="MSG-TW analytic")
        a.loglog(hw, [res[h]["RM_ctr"][idx, idx] for h in HW], "-o", color="#1f77b4", label="RM, centre")
        a.loglog(hw, [res[h]["KF_ctr"][idx, idx] for h in HW], "-s", color="#2ca02c", label="KF, centre")
        a.loglog(hw, [res[h]["RM_OML"][idx, idx] for h in HW], "--^", color="#ff7f0e", label="RM, OML")
        a.loglog(hw, [res[h]["KF_OML"][idx, idx] for h in HW], "--v", color="#d62728", label="KF, OML")
        a.set_title(f"${{{cij[0]}}}_{{{cij[1:]}}}$ (${eng}$)" if False else f"{cij} ({eng})")
        a.set_xlabel("$h/W$"); a.set_ylabel(f"{eng}  [SI]")
        a.grid(True, which="both", ls=":", alpha=0.5)
    ax.flat[0].legend(fontsize=11, loc="best")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(REP, "strip_iso_abs.png"), dpi=150); plt.close(fig)

    # ---- plot 2: % error vs the solid ----
    fig, ax = plt.subplots(3, 2, figsize=(13, 15))
    fig.suptitle("Isotropic strip: shell-vs-solid \\% error (FEniCS-solid = 0 line)",
                 fontweight="bold")
    for k, (idx, cij, eng) in enumerate(DIAG):
        a = ax.flat[k]
        sv = np.array([res[h]["solid"][idx, idx] for h in HW])
        for lbl, key, col, ls, mk in [("RM, centre", "RM_ctr", "#1f77b4", "-", "o"),
                                       ("KF, centre", "KF_ctr", "#2ca02c", "-", "s"),
                                       ("RM, OML", "RM_OML", "#ff7f0e", "--", "^"),
                                       ("KF, OML", "KF_OML", "#d62728", "--", "v")]:
            ys = np.array([100*(res[h][key][idx, idx]-res[h]["solid"][idx, idx])
                           / res[h]["solid"][idx, idx] for h in HW])
            a.plot(hw, ys, color=col, ls=ls, marker=mk, label=lbl)
        ya = np.array([100*(res[h]["analytic"][cij]-res[h]["solid"][idx, idx])
                       / res[h]["solid"][idx, idx] for h in HW])
        a.plot(hw, ya, ":", color="0.1", marker="P", label="MSG-TW analytic")
        fin = ya[np.isfinite(ya)]
        allv = np.concatenate([fin[np.abs(fin) <= 100],
                               [100*(res[h][key][idx, idx]-res[h]["solid"][idx, idx])
                                / res[h]["solid"][idx, idx] for h in HW for key in
                                ("RM_ctr", "KF_ctr", "RM_OML", "KF_OML")]])
        allv = allv[np.abs(allv) <= 100]
        M = max(3.0, 1.25*np.max(np.abs(allv))) if len(allv) else 30.0
        a.set_ylim(-M, M); a.axhline(0, color="0.45", lw=1.0)
        a.set_title(f"{cij} ({eng})"); a.set_xlabel("$h/W$"); a.set_ylabel("\\% error vs solid")
        a.grid(True, ls=":", alpha=0.6)
    ax.flat[0].legend(fontsize=11, loc="best")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(REP, "strip_iso_err.png"), dpi=150); plt.close(fig)
    print("\nwrote report/strip_iso_table.txt, strip_iso_abs.png, strip_iso_err.png")


if __name__ == "__main__":
    main()
