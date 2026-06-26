"""
Anisotropic circular tube (Opensg_MSG Table 3.2/3.3): single -45 deg ply,
R=0.0715 m, h=0.008682 m; E1=37 E2=E3=9 G=4 GPa, nu=0.3.

  1. generate the tube YAML (nodes at OML, frac=0.5 -> centroid reference = the
     document's centre reference; plate ABD and geometry share that reference);
  2. homogenize and validate vs Table 3.2 (classical 4x4, incl. tension-torsion C12);
  3. plot e2 (tangent) and e3 (inward = layup-stacking) on the section;
  4. dehom under a realistic load, stress through the wall on the path y2=0, y3>0.

(FEniCS-solid dehom comparison = next step: needs the solid annulus mesh + run.)
"""
import os, sys
import numpy as np
import yaml as _yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import solve_tw_from_yaml, stress_at_points
from fe_jax.msg_mesh import element_e3_from_yaml

R, H, ANG, N = 0.0715, 0.008682, -45.0, 120
OUT = os.path.join(HERE, "..", "outputs", "tube_dehom")
YAML = os.path.join(OUT, "aniso_tube.yaml")
TAB = {"C11 EA": 47.785e6, "C12 ext-tw": -0.93755e6, "C22 GJ": 0.14896e6,
       "C33 EI": 0.10710e6}        # Table 3.2 (centre ref)
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]


def gen_yaml():
    os.makedirs(OUT, exist_ok=True)
    Rg = R + H/2.0                                   # OML radius (frac=0.5 -> R)
    th = np.array([2*np.pi*k/N for k in range(N)])
    nodes = [[float(Rg*np.cos(t)), float(Rg*np.sin(t)), 0.0] for t in th]
    elements = [[k+1, k+2] for k in range(N-1)] + [[N, 1]]
    thm = np.array([np.pi*(2*k+1)/N for k in range(N)])
    ori = [[0., 0., 1., float(-np.sin(t)), float(np.cos(t)), 0.,
            float(-np.cos(t)), float(-np.sin(t)), 0.] for t in thm]  # e3 inward
    data = {"nodes": nodes, "elements": elements,
            "sets": {"element": [{"name": "tube", "labels": list(range(1, N+1))}]},
            "sections": [{"elementSet": "tube", "layup": [["aniso", H, ANG]]}],
            "materials": [{"name": "aniso", "density": 1800.0,
                           "elastic": {"E": [37e9, 9e9, 9e9], "G": [4e9, 4e9, 4e9],
                                       "nu": [0.3, 0.3, 0.3]}}],
            "elementOrientations": ori}
    with open(YAML, "w") as f:
        _yaml.safe_dump(data, f)


def plot_e2e3(bundle):
    corners = np.asarray(bundle["corners"]); rc = np.asarray(bundle["red_cells"])
    xd2 = np.asarray(bundle["xd2"]); xd3 = np.asarray(bundle["xd3"])
    e3 = element_e3_from_yaml(YAML); cen = corners.mean(0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(corners[:, 0], corners[:, 1], ".", ms=2, color="0.6")
    sc = 0.10*R
    for e in range(rc.shape[0]):
        if e % 4: continue
        mid = 0.5*(corners[int(rc[e, 0])]+corners[int(rc[e, 1])])
        t = np.array([xd2[e], xd3[e]])                    # e2 = tangent
        ax.arrow(mid[0], mid[1], sc*t[0], sc*t[1], head_width=sc*0.2,
                 color="tab:blue", length_includes_head=True)
        ax.arrow(mid[0], mid[1], sc*e3[e, 0], sc*e3[e, 1], head_width=sc*0.2,
                 color="tab:red", length_includes_head=True)
    ax.plot([], [], color="tab:blue", label="e2 (tangent)")
    ax.plot([], [], color="tab:red", label="e3 (inward = layup stacking)")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Anisotropic tube: material frame e2/e3 (fiber {ANG:g}deg in e1-e2)")
    ax.set_xlabel("y2 (m)"); ax.set_ylabel("y3 (m)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "e2_e3_layup.png"), dpi=150)
    plt.close(fig); print("wrote", os.path.join(OUT, "e2_e3_layup.png"))


def main():
    gen_yaml()
    b = solve_tw_from_yaml(YAML, frac=0.5)               # centroid ref (= Table 3.2)
    EB = np.asarray(b["EB"])
    print("Anisotropic tube homogenization (centroid ref) vs Table 3.2:\n")
    print(f"  {'term':12s}{'MSG':>13s}{'Table 3.2':>13s}{'% err':>9s}")
    for nm, (i, j) in [("C11 EA", (0, 0)), ("C12 ext-tw", (0, 1)),
                       ("C22 GJ", (1, 1)), ("C33 EI", (2, 2))]:
        v = TAB[nm]
        print(f"  {nm:12s}{EB[i,j]:13.4e}{v:13.4e}{100*(EB[i,j]-v)/v:9.1f}")

    plot_e2e3(b)

    # ---- realistic load + dehom on y2=0, y3>0 ----
    FF = np.array([2.0e4, 0.0, 0.0, 3.0e2, 1.5e2, 0.0])   # [F1,F2,F3,M1,M2,M3]
    nz = 15
    yy = np.linspace(R + H/2, R - H/2, nz)               # OML->IML at the top
    coords = np.column_stack([np.zeros(nz), yy])
    out = stress_at_points(b, coords, beam_force_vabs=FF, frame="material")
    S = out["stress"]
    zt = (R + H/2 - yy)*1e3                               # depth mm
    print(f"\nDehom on y2=0,y3>0 (top wall, OML->IML), load {FF.tolist()}:")
    print("   depth(mm)  " + "  ".join(f"{c:>9s}" for c in COMP) + "  (MPa)")
    for k in range(nz):
        print(f"   {zt[k]:7.2f} " + "  ".join(f"{S[k,j]/1e6:9.3f}" for j in range(6)))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Anisotropic tube {ANG:g}deg, dehom on y2=0,y3>0 (material frame)",
                 fontweight="bold")
    for j, c in enumerate(COMP):
        a = ax.flat[j]; a.plot(zt, S[:, j]/1e6, "r-o", ms=4)
        a.set_title(f"$\\sigma_{{{c[1:]}}}$"); a.set_xlabel("depth (mm, OML->IML)")
        a.set_ylabel(f"{c} (MPa)"); a.grid(True, ls=":", alpha=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "tube_dehom_top.png"), dpi=150); plt.close(fig)
    print("wrote", os.path.join(OUT, "tube_dehom_top.png"))

    # ---- equilibrium self-check: recovered global sigma_11 integrates to F1,M2,M3
    thm = np.array([np.pi*(2*k+1)/N for k in range(N)])
    pe = np.column_stack([R*np.cos(thm), R*np.sin(thm)])     # mid-surface, all elems
    s11 = stress_at_points(b, pe, beam_force_vabs=FF, frame="global")["stress"][:, 0]
    ds = 2*np.pi*R/N
    F1r = float(np.sum(s11*H*ds)); M2r = float(np.sum(s11*pe[:, 1]*H*ds))
    M3r = float(np.sum(s11*pe[:, 0]*H*ds))
    print("\nEquilibrium self-check (global frame, recovered sigma_11 around section):")
    print(f"   F1: applied {FF[0]:11.2f}   recovered {F1r:11.2f}   err {100*(F1r-FF[0])/FF[0]:+.2f}%")
    print(f"   M2: applied {FF[4]:11.2f}   recovered {abs(M2r):11.2f}   err {100*(abs(M2r)-FF[4])/FF[4]:+.2f}%")
    print(f"   M3: applied {FF[5]:11.2f}   recovered {abs(M3r):11.2f}   (applied 0)")


if __name__ == "__main__":
    main()
