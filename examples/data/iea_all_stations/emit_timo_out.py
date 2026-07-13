'''
emit_timo_out.py  --  collate the Timoshenko 6x6 STIFFNESS + COMPLIANCE (+ wall-time) per span station
                      into a labelled .OUT file, one per OpenSG source (like VABS prints [K] and inv[K]).
================================================================================================
Reads the per-station 6x6 stiffness .txt written by the three OpenSG homogenizers (+ their
times_<key>.csv if present) and writes ONE .out per source, each block headed by WHERE the 6x6 came
from + the sign convention + the homogenization wall-time, then the 6x6 stiffness [K] and its
compliance [S] = inv(K):

    homo_rm/     C6_rm_<name>.txt      (+ times_rm.csv)      -> out/timo_rm.out       (RM SHELL,  OpenSG-RM)
    homo_jax/    C6_jax_<name>.txt     (+ times_jax.csv)     -> out/timo_jax.out      (2-D SOLID, OpenSG-JAX)
    homo_fenics/ C6_fenics_<name>.txt  (+ times_fenics.csv)  -> out/timo_fenics.out   (2-D SOLID, OpenSG-FEniCSx)

File extension: .out = a RESULT/output file (the homogenization output), analogous to VABS's output;
reserve .dat for the GEBT INPUT file (which GEBT reads).
CONVENTION (VABS/OpenSG order): 1=extension, 2-3=transverse shear, 4=torsion, 5-6=bending.

    python emit_timo_out.py
================================================================================================
'''
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {
    "rm":     ("homo_rm",     "C6_rm_",     "Timoshenko 6x6 from RM SHELL cross-section  --  OpenSG-RM (6-DOF drilling-Lagrange ring, MITC gamma_23)"),
    "jax":    ("homo_jax",    "C6_jax_",    "Timoshenko 6x6 from 2-D SOLID cross-section  --  OpenSG-JAX (JAX-FEM Mechanics of Structure Genome)"),
    "fenics": ("homo_fenics", "C6_fenics_", "Timoshenko 6x6 from 2-D SOLID cross-section  --  OpenSG-FEniCSx (dolfinx FE Mechanics of Structure Genome)"),
}
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def tag_to_r(tag):
    m = re.search(r"r(\d{4})", tag)
    return int(m.group(1)) / 1000.0 if m else float("nan")


def load_times(folder, key):
    """times_<key>.csv: '<tag>,<seconds>' per line -> {tag: seconds}."""
    p = os.path.join(HERE, folder, "times_%s.csv" % key)
    t = {}
    if os.path.exists(p):
        for ln in open(p):
            parts = ln.strip().split(",")
            if len(parts) >= 2:
                try:
                    t[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return t


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
    times = load_times(folder, key)
    ttot = sum(times.values()) if times else float("nan")
    with open(out, "w") as f:
        f.write("#" + "=" * 96 + "\n")
        f.write("# %s\n" % label)
        f.write("# convention (VABS/OpenSG order): 1=extension, 2-3=transverse shear, 4=torsion, 5-6=bending\n")
        f.write("#                                 diag([K]) = [EA, GA2, GA3, GJ, EI2, EI3];  units SI\n")
        f.write("# each station lists the homogenization wall-time, the 6x6 stiffness [K], and [S]=inv([K])\n")
        f.write("# stations: %d    total homogenization wall-time: %s\n"
                % (len(files), ("%.1f s" % ttot) if ttot == ttot else "n/a"))
        f.write("#" + "=" * 96 + "\n\n")
        for fp in files:
            tag = os.path.basename(fp).replace(prefix, "").replace(".txt", "")
            r = tag_to_r(tag)
            K = np.loadtxt(fp)
            try:
                S = np.linalg.inv(K)
            except Exception:
                S = np.full((6, 6), np.nan)
            tt = times.get(tag)
            f.write("=============== station %-12s  r = %.4f ===============\n" % (tag, r))
            f.write("source: %s\n" % label.split("--")[-1].strip())
            f.write("homogenization wall-time: %s\n" % (("%.2f s" % tt) if tt is not None else "n/a"))
            f.write("diag([K]) [EA GA2 GA3 GJ EI2 EI3] = %s\n\n" % "  ".join("%.6e" % K[i, i] for i in range(6)))
            mat(f, "Stiffness [K]", K)
            f.write("\n")
            mat(f, "Compliance [S] = inv([K])", S)
            f.write("\n")
    print("wrote %s  (%d stations, total %s)" % (out, len(files), ("%.1f s" % ttot) if ttot == ttot else "n/a"))


def main():
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    for key, (folder, prefix, label) in SRC.items():
        emit(key, folder, prefix, label, os.path.join(out, "timo_%s.out" % key))


if __name__ == "__main__":
    main()
