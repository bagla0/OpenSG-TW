'''
emit_timo_dat.py  --  collate the Timoshenko 6x6 STIFFNESS + COMPLIANCE per span station into a .dat,
                      LABELLED by its OpenSG source (like VABS prints [K] and its inverse [S]).
================================================================================================
Reads the per-station 6x6 Timoshenko stiffness .txt written by the three OpenSG homogenizers and
writes ONE .dat per source, each block headed by WHERE the 6x6 came from + the sign convention, then
the 6x6 stiffness [K] and its compliance [S] = inv(K):

    homo_rm/C6_rm_<name>.txt         ->  dat/timo_rm.dat       (RM SHELL,   OpenSG-RM 6-DOF ring)
    homo_jax/C6_jax_<name>.txt       ->  dat/timo_jax.dat      (2-D SOLID,  OpenSG-JAX  FE-MSG)
    homo_fenics/C6_fenics_<name>.txt ->  dat/timo_fenics.dat   (2-D SOLID,  OpenSG-FEniCSx FE-MSG)

CONVENTION (VABS / OpenSG order): 1 = extension, 2-3 = transverse shear, 4 = torsion, 5-6 = bending
=> diag(K) = [EA, GA2, GA3, GJ, EI2, EI3].   Units SI.

    python emit_timo_dat.py
================================================================================================
'''
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# key : (folder, file-prefix, human source label written before every 6x6)
SRC = {
    "rm":     ("homo_rm",     "C6_rm_",     "Timoshenko 6x6 from RM SHELL cross-section  --  OpenSG-RM (6-DOF drilling-Lagrange ring, MITC gamma_23)"),
    "jax":    ("homo_jax",    "C6_jax_",    "Timoshenko 6x6 from 2-D SOLID cross-section  --  OpenSG-JAX (JAX-FEM Mechanics of Structure Genome)"),
    "fenics": ("homo_fenics", "C6_fenics_", "Timoshenko 6x6 from 2-D SOLID cross-section  --  OpenSG-FEniCSx (dolfinx FE Mechanics of Structure Genome)"),
}
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def tag_to_r(tag):
    m = re.search(r"r(\d{4})", tag)
    return int(m.group(1)) / 1000.0 if m else float("nan")


def mat(f, name, M):
    f.write("%s:\n" % name)
    for i in range(6):
        f.write("  " + "  ".join("% .10e" % M[i, j] for j in range(6)) + "\n")


def emit(key, folder, prefix, label, out):
    files = sorted(glob.glob(os.path.join(HERE, folder, prefix + "*.txt")),
                   key=lambda p: tag_to_r(os.path.basename(p)))
    if not files:
        print("skip %-7s (no %s/%s*.txt yet)" % (key, folder, prefix))
        return
    with open(out, "w") as f:
        f.write("#" + "=" * 96 + "\n")
        f.write("# %s\n" % label)
        f.write("# convention (VABS/OpenSG order): 1=extension, 2-3=transverse shear, 4=torsion, 5-6=bending\n")
        f.write("#                                 diag([K]) = [EA, GA2, GA3, GJ, EI2, EI3];  units SI\n")
        f.write("# each station lists the 6x6 stiffness [K] and its compliance [S] = inv([K])\n")
        f.write("# stations: %d\n" % len(files))
        f.write("#" + "=" * 96 + "\n\n")
        for fp in files:
            tag = os.path.basename(fp).replace(prefix, "").replace(".txt", "")
            r = tag_to_r(tag)
            K = np.loadtxt(fp)
            try:
                S = np.linalg.inv(K)
            except Exception:
                S = np.full((6, 6), np.nan)
            f.write("=============== station %-12s  r = %.4f  (%s) ===============\n" % (tag, r, label.split("--")[-1].strip()))
            f.write("diag([K]) [EA GA2 GA3 GJ EI2 EI3] = %s\n\n" % "  ".join("%.6e" % K[i, i] for i in range(6)))
            mat(f, "Stiffness [K]", K)
            f.write("\n")
            mat(f, "Compliance [S] = inv([K])", S)
            f.write("\n")
    print("wrote %s  (%d stations, source: %s)" % (out, len(files), label.split("--")[-1].strip()))


def main():
    out = os.path.join(HERE, "dat")
    os.makedirs(out, exist_ok=True)
    for key, (folder, prefix, label) in SRC.items():
        emit(key, folder, prefix, label, os.path.join(out, "timo_%s.dat" % key))


if __name__ == "__main__":
    main()
