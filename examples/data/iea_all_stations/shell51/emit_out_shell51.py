'''
emit_out_shell51.py  --  write one Timoshenko 6x6 .out per station into shell51/out/
================================================================================================
Dedicated emitter for the 51-station shell batch.  Matches the EXACT format of the base-dir
emit_timo_out.py (which is USER-EDITED and must NOT be modified) for the RM-Shell source, but:
  - reads shell51/homo_rm/OpenSG_RM_iea_sNN.txt          (the 6x6 stiffness homo_rm_shell.py wrote)
  - parses per-station wall-time from shell51/log_rm.txt (lines "[iea_sNN ] ... [X.Xs]")
  - writes shell51/out/iea_sNN_OpenSG_RM_Shell.out       (flat, no per-source subfolder)

Each .out = source label + VABS/OpenSG convention + Time-taken + [K] (Stiffness) + inv(K) (Compliance).
RM has no mass matrix -> no Mass block.

    python emit_out_shell51.py
================================================================================================
'''
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = "Timoshenko 6x6 -- RM SHELL cross-section (OpenSG-RM, Boundary ring, MITC gamma_23)"
PREFIX = "OpenSG_RM_"          # homo_rm_shell.py writes OpenSG_RM_<name>.txt
SUFFIX = "OpenSG_RM_Shell"


def load_times(log_path):
    """Parse '[iea_sNN   ] ... [X.Xs]' lines -> {tag: seconds}."""
    t = {}
    if not os.path.exists(log_path):
        return t
    pat = re.compile(r"^\[\s*(\S+)\s*\].*\[([\d.]+)s\]\s*$")
    for ln in open(log_path):
        m = pat.match(ln.strip())
        if m:
            try:
                t[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return t


def mat(f, name, M):
    f.write("%s:\n" % name)
    for i in range(6):
        f.write("  " + "  ".join("% .10e" % M[i, j] for j in range(6)) + "\n")


def main():
    homo = os.path.join(HERE, "homo_rm")
    outdir = os.path.join(HERE, "out")
    os.makedirs(outdir, exist_ok=True)
    times = load_times(os.path.join(HERE, "log_rm.txt"))
    files = sorted(glob.glob(os.path.join(homo, PREFIX + "*.txt")))
    if not files:
        print("no %s/%s*.txt found" % (homo, PREFIX))
        return
    n = 0
    for fp in files:
        tag = os.path.basename(fp).replace(PREFIX, "").replace(".txt", "")   # e.g. iea_s12
        K = np.loadtxt(fp)
        try:
            S = np.linalg.inv(K)
        except Exception:
            S = np.full((6, 6), np.nan)
        tt = times.get(tag)
        with open(os.path.join(outdir, "%s_%s.out" % (tag, SUFFIX)), "w") as f:
            f.write("# %s\n" % LABEL)
            f.write("# convention (VABS/OpenSG order): 1=extension, 2-3=transverse shear, 4=torsion, "
                    "5-6=bending\n")
            f.write("# Time-taken: %s\n\n" % (("%.2f s" % tt) if tt is not None else "n/a"))
            mat(f, "Stiffness ", K)
            f.write("\n")
            mat(f, "Compliance", S)
        n += 1
    print("wrote %d x *_%s.out -> %s" % (n, SUFFIX, outdir))


if __name__ == "__main__":
    main()
