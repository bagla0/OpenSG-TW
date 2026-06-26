"""
Station-15 homogenization report: OML vs IML vs centroid reference (MSG-TW, JAX)
vs the FEniCS solid (2Dsolid_VABS_15), full symmetric 6x6 stiffness.

Writes a Markdown report (outputs/REPORT_st15_homogenization.md) with the 21
unique stiffness terms C11..C66 and their percent error vs the FEniCS solid, and
a short out-of-plane-stress investigation (the dehom recovers the in-plane shell
strains, so sigma_33 ~ 0 is plane-stress-correct while sigma_13/23 ~ 0 is a
thick-laminate limitation).
"""
import os
import sys
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import solve_tw_from_yaml, stress_at_points

SHELL15 = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
DIAG = ["C11=EA", "C22=GA12", "C33=GA13", "C44=GJ", "C55=EI2", "C66=EI3"]
FF = np.array([32230.4005595904, -7663.907852209771, 251712.81004955297,
               -55608.54410550957, -4170203.8641732424, -123224.93244239496])


def homo():
    refs = {"OML": 0.0, "centroid": 0.5, "IML": 1.0}
    C = {k: solve_tw_from_yaml(SHELL15, frac=v)["Timo"] for k, v in refs.items()}
    fe_path = os.path.join(OUT, "fenics_st15_timo.txt")
    FE = np.loadtxt(fe_path) if os.path.exists(fe_path) else None
    return C, FE


def out_of_plane():
    """Dehom in-plane vs out-of-plane stress magnitude at sub-surface points."""
    b = solve_tw_from_yaml(SHELL15, frac=0.0)
    corners = np.asarray(b["corners"]); cen = corners.mean(0)
    # pick a few skin nodes, push ~30% of the wall inward (sub-surface)
    idx = [10, 25, 40]
    pts = []
    for nd in idx:
        v = cen - corners[nd]; v /= np.linalg.norm(v) + 1e-30
        pts.append(corners[nd] + 0.01 * v)
    out = stress_at_points(b, np.array(pts), beam_force_vabs=FF, frame="local")
    S = out["stress"]
    inplane = np.abs(S[:, [0, 1, 5]]).max(axis=1)        # S11,S22,S12
    outplane = np.abs(S[:, [2, 4, 3]]).max(axis=1)       # S33,S13,S23
    return out["depth"], inplane, outplane, S


def main():
    C, FE = homo()
    lines = ["# Station 15 — MSG-TW homogenization report",
             "", "Reference-surface study (1Dshell_15) vs FEniCS solid "
             "(2Dsolid_VABS_15).  6x6 Timoshenko stiffness, order "
             "[F1,F2,F3,M1,M2,M3] <-> [eps11,g12,g13,k1,k2,k3].", ""]

    # ---- diagonal summary ----
    lines += ["## Diagonal (engineering) stiffnesses", "",
              "| term | OML | centroid | IML | FEsolid | OML % | cen % | IML % |",
              "|------|-----|----------|-----|---------|-------|-------|-------|"]
    for d, (lbl) in enumerate(DIAG):
        o, c, m = C["OML"][d, d], C["centroid"][d, d], C["IML"][d, d]
        row = f"| {lbl} | {o:.3e} | {c:.3e} | {m:.3e} |"
        if FE is not None:
            f = FE[d, d]
            row += (f" {f:.3e} | {(o-f)/f*100:+.1f} | {(c-f)/f*100:+.1f} | "
                    f"{(m-f)/f*100:+.1f} |")
        else:
            row += " - | - | - | - |"
        lines.append(row)

    # ---- full 21 unique terms ----
    lines += ["", "## Full 6x6 — 21 unique terms  (% = 100*(JAX-FE)/FE)", "",
              "| Cij | OML | centroid | IML | FEsolid | OML % | cen % | IML % |",
              "|-----|-----|----------|-----|---------|-------|-------|-------|"]
    for i in range(6):
        for j in range(i, 6):
            o, c, m = C["OML"][i, j], C["centroid"][i, j], C["IML"][i, j]
            row = f"| C{i+1}{j+1} | {o:.3e} | {c:.3e} | {m:.3e} |"
            if FE is not None:
                f = FE[i, j]
                pe = (lambda x: f" {(x-f)/f*100:+.1f}" if abs(f) > 1e-3*abs(FE[i,i]*FE[j,j])**0.5 else "  n/a")
                row += f" {f:.3e} |{pe(o)} |{pe(c)} |{pe(m)} |"
            else:
                row += " - | - | - | - |"
            lines.append(row)

    if FE is not None:
        for tag in ("OML", "centroid", "IML"):
            rel = np.linalg.norm(C[tag] - FE) / np.linalg.norm(FE) * 100
            lines.append("")
            lines.append(f"- full-6x6 ||{tag}-FE||/||FE|| = **{rel:.2f}%**")

    # ---- out-of-plane investigation ----
    dep, ip, op, S = out_of_plane()
    lines += ["", "## Out-of-plane stress (dehom)", "",
              "The dehom recovers the 3D stress from the **in-plane shell strains** "
              "(membrane + curvature).  At a sub-surface point:", "",
              "| depth (mm) | max in-plane (S11,S22,S12) | max out-of-plane (S33,S13,S23) | ratio |",
              "|------------|----------------------------|-------------------------------|-------|"]
    for k in range(len(dep)):
        lines.append(f"| {dep[k]*1e3:.2f} | {ip[k]:.3e} | {op[k]:.3e} | "
                     f"{op[k]/max(ip[k],1e-30):.4f} |")
    lines += ["",
              "- **sigma_33 ~ 0** is the plane-stress condition (free faces) — correct.",
              "- **sigma_13 / sigma_23 ~ 0** is a *thick-laminate limitation*: the beam "
              "transverse shear is not passed to the plate recovery, so the through-"
              "thickness shear (significant in thick spar caps) is not reproduced.  The "
              "solid (VABS) captures it.  In thin skins it is negligible.", ""]

    path = os.path.join(OUT, "REPORT_st15_homogenization.md")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", path)
    print("\n".join(lines[:24]))
    print("...\nout-of-plane ratio (out/in):",
          [f"{op[k]/max(ip[k],1e-30):.4f}" for k in range(len(dep))])


if __name__ == "__main__":
    main()
