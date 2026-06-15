"""
Compute the 6x6 Timoshenko (and 4x4 Euler-Bernoulli) beam stiffness of a
thin-walled composite cross-section from an OpenSG Shell_1DSG YAML file, using
the MSG Hermite-C1 shell homogenization (no FEniCSx / MPI).

Usage
-----
    python compute_timoshenko.py [<cross_section.yaml>] [<reference>]

    <cross_section.yaml>  OpenSG Shell_1DSG YAML (default: 1Dshell_15.yaml)
    <reference>           OML (default) | CENTROID | IML  -- reference surface

Output
------
Writes outputs/homo/<yaml_stem>_<reference>_timo.txt with the 6x6 Timoshenko
matrix (VABS order [F1,F2,F3,M1,M2,M3] <-> [ext, shear2, shear3, twist,
bend2, bend3]), the 4x4 Euler-Bernoulli matrix, and the engineering diagonals.
"""
import os
import sys
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "opensg_jax"))
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax import timoshenko_from_yaml

DEFAULT_YAML = r"C:\Users\bagla0\OpenSG\examples\data\Shell_1DSG\1Dshell_15.yaml"
OUT_DIR = os.path.join(HERE, "..", "..", "outputs", "homo")
DIAG = ["C11 = EA   (extension)", "C22 = GA22 (shear 2)", "C33 = GA33 (shear 3)",
        "C44 = GJ   (torsion)", "C55 = EI2  (bending 2)", "C66 = EI3  (bending 3)"]


def _fmt(M):
    return "\n".join("  " + "  ".join(f"{v: .8e}" for v in row) for row in M)


def main():
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_YAML
    reference = (sys.argv[2] if len(sys.argv) > 2 else "OML").upper()
    if not os.path.exists(yaml_path):
        sys.exit(f"YAML not found: {yaml_path}")

    EB, Timo, complete = timoshenko_from_yaml(yaml_path, reference=reference)
    EB = np.asarray(EB); Timo = np.asarray(Timo)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(yaml_path))[0]
    out_path = os.path.join(OUT_DIR, f"{stem}_{reference}_timo.txt")

    lines = [
        f"# MSG Hermite-C1 shell homogenization (OpenSG-TWJAX)",
        f"# cross-section : {yaml_path}",
        f"# reference     : {reference}   (all elements used: {complete})",
        "",
        "# 6x6 Timoshenko stiffness",
        "# order [F1,F2,F3,M1,M2,M3] <-> [ext, shear2, shear3, twist, bend2, bend3]",
        _fmt(Timo),
        "",
        "# 4x4 Euler-Bernoulli stiffness   order [ext, twist, bend2, bend3]",
        _fmt(EB),
        "",
        "# engineering diagonals (Timoshenko)",
    ]
    for i, lab in enumerate(DIAG):
        lines.append(f"  {lab:28s} = {Timo[i, i]: .8e}")
    text = "\n".join(lines) + "\n"

    with open(out_path, "w") as f:
        f.write(text)

    print(text)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
