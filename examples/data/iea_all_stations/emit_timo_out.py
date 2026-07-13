'''
emit_timo_out.py  --  write ONE .out file PER STATION PER OpenSG source, holding that station's
                      Timoshenko 6x6 stiffness [K] + compliance [S]=inv(K) + homogenization wall-time.
================================================================================================
For every per-station 6x6 written by the three OpenSG homogenizers it emits, into out/ :

    <xml-name>_OpenSG_RM_Shell.out       (from homo_rm/C6_rm_<name>.txt      + times_rm.csv)
    <xml-name>_OpenSG_JAX_Solid.out      (from homo_jax/C6_jax_<name>.txt    + times_jax.csv)
    <xml-name>_OpenSG_FEniCSx_Solid.out  (from homo_fenics/C6_fenics_<name>  + times_fenics.csv)

e.g. iea_r0247_OpenSG_RM_Shell.out .  The station is in the FILENAME, so it is NOT repeated inside;
each file is self-contained (source label + VABS convention + wall-time + [K] + [S]).  .out = a
RESULT file (like VABS's output); .dat is reserved for the GEBT INPUT.
CONVENTION (VABS/OpenSG order): 1=extension, 2-3=transverse shear, 4=torsion, 5-6=bending.

    python emit_timo_out.py
================================================================================================
'''
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# key : (folder, file-prefix, filename-suffix, in-file source label)
SRC = {
    "rm":     ("homo_rm",     "C6_rm_",     "OpenSG_RM_Shell",
               "Timoshenko 6x6 -- RM SHELL cross-section (OpenSG-RM, 6-DOF drilling-Lagrange ring, MITC gamma_23)"),
    "jax":    ("homo_jax",    "C6_jax_",    "OpenSG_JAX_Solid",
               "Timoshenko 6x6 -- 2-D SOLID cross-section (OpenSG-JAX, JAX-FEM Mechanics of Structure Genome)"),
    "fenics": ("homo_fenics", "C6_fenics_", "OpenSG_FEniCSx_Solid",
               "Timoshenko 6x6 -- 2-D SOLID cross-section (OpenSG-FEniCSx, dolfinx FE Mechanics of Structure Genome)"),
}
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def load_times(folder, key):
    p = os.path.join(HERE, folder, "times_%s.csv" % key)
    t = {}
    if os.path.exists(p):
        for ln in open(p):
            a = ln.strip().split(",")
            if len(a) >= 2:
                try:
                    t[a[0]] = float(a[1])
                except ValueError:
                    pass
    return t


def mat(f, name, M):
    f.write("%s:\n" % name)
    for i in range(6):
        f.write("  " + "  ".join("% .10e" % M[i, j] for j in range(6)) + "\n")


def emit(key, folder, prefix, suffix, label, outdir):
    files = sorted(glob.glob(os.path.join(HERE, folder, prefix + "*.txt")))
    if not files:
        print("skip %-7s (no %s/%s*.txt)" % (key, folder, prefix))
        return 0
    times = load_times(folder, key)
    for fp in files:
        tag = os.path.basename(fp).replace(prefix, "").replace(".txt", "")     # e.g. iea_r0247
        K = np.loadtxt(fp)
        try:
            S = np.linalg.inv(K)
        except Exception:
            S = np.full((6, 6), np.nan)
        tt = times.get(tag)
        with open(os.path.join(outdir, "%s_%s.out" % (tag, suffix)), "w") as f:
            f.write("# %s\n" % label)
            f.write("# convention (VABS/OpenSG order): 1=extension, 2-3=transverse shear, 4=torsion, "
                    "5-6=bending;  diag([K]) = [EA, GA2, GA3, GJ, EI2, EI3];  units SI\n")
            f.write("# homogenization wall-time: %s\n\n" % (("%.2f s" % tt) if tt is not None else "n/a"))
            f.write("diag([K]) [EA GA2 GA3 GJ EI2 EI3] = %s\n\n"
                    % "  ".join("%.6e" % K[i, i] for i in range(6)))
            mat(f, "Stiffness [K]", K)
            f.write("\n")
            mat(f, "Compliance [S] = inv([K])", S)
    print("wrote %d x *_%s.out" % (len(files), suffix))
    return len(files)


def main():
    outdir = os.path.join(HERE, "out")
    os.makedirs(outdir, exist_ok=True)
    n = 0
    for key, (folder, prefix, suffix, label) in SRC.items():
        n += emit(key, folder, prefix, suffix, label, outdir)
    print("total %d per-station .out files -> %s" % (n, outdir))


if __name__ == "__main__":
    main()
