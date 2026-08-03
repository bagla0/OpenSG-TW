'''
verify_shell51.py  --  sanity-check the 51-station RM 6x6 batch
================================================================================================
 * prints the Timoshenko diagonal (EA, GA2, GA3, GJ, EI2, EI3) vs eta for every station,
 * flags any WILD station (non-positive diagonal, or a >10x jump vs its neighbour),
 * cross-checks iea_s12 (eta=0.24) against the existing reference r0247 (eta=0.24696).
================================================================================================
'''
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
L_BLADE = 138.204


def read_K_from_out(path):
    """Parse the 6x6 'Stiffness :' block from a *.out file."""
    lines = open(path).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Stiffness"):
            rows = []
            for j in range(i + 1, i + 7):
                rows.append([float(x) for x in lines[j].split()])
            return np.array(rows)
    raise ValueError("no Stiffness block in %s" % path)


def main():
    homo = os.path.join(HERE, "homo_rm")
    diag = {}
    for i in range(51):
        tag = "iea_s%02d" % i
        fp = os.path.join(homo, "OpenSG_RM_%s.txt" % tag)
        if os.path.exists(fp):
            K = np.loadtxt(fp)
            diag[i] = (i / 50.0, np.diag(K), K)

    print("=== Timoshenko diagonal vs eta (all %d stations) ===" % len(diag))
    print("%-8s %6s | %11s %11s %11s %11s %11s %11s" % ("name", "eta", *LBL))
    for i in sorted(diag):
        eta, d, _ = diag[i]
        print("iea_s%02d  %6.3f | %s" % (i, eta, " ".join("%11.4e" % v for v in d)))

    # ---- wild-station flags -------------------------------------------------
    print("\n=== wild-station flags ===")
    flags = []
    keys = sorted(diag)
    for k in keys:
        eta, d, _ = diag[k]
        for j in range(6):
            if not np.isfinite(d[j]) or d[j] <= 0:
                flags.append("iea_s%02d %s = %.4e (non-positive/nan)" % (k, LBL[j], d[j]))
    for a, b in zip(keys[:-1], keys[1:]):
        da = diag[a][1]; db = diag[b][1]
        for j in range(6):
            if da[j] > 0 and db[j] > 0:
                r = db[j] / da[j]
                if r > 10 or r < 0.1:
                    flags.append("iea_s%02d->iea_s%02d %s jumps x%.1f" % (a, b, LBL[j], r))
    if flags:
        for fl in flags:
            print("  FLAG:", fl)
    else:
        print("  none -- all diagonals positive, finite, and within 10x station-to-station")

    # ---- ~6 sample stations for the smooth-variation summary ----------------
    print("\n=== smooth-variation samples (eta ~ 0, .2, .4, .6, .8, 1) ===")
    for i in [0, 10, 20, 30, 40, 50]:
        if i in diag:
            eta, d, _ = diag[i]
            print("iea_s%02d eta=%.2f z=%6.2fm : EA=%.4e GA2=%.4e GA3=%.4e GJ=%.4e EI2=%.4e EI3=%.4e"
                  % (i, eta, eta * L_BLADE, d[0], d[1], d[2], d[3], d[4], d[5]))

    # ---- cross-check iea_s12 (eta 0.24) vs reference r0247 (eta 0.24696) ------
    print("\n=== cross-check: iea_s12 (eta 0.2400) vs reference iea_r0247 (eta 0.24696) ===")
    ref = os.path.join(BASE, "out", "OpenSG_RM_Shell", "iea_r0247_OpenSG_RM_Shell.out")
    s12 = os.path.join(homo, "OpenSG_RM_iea_s12.txt")
    if os.path.exists(ref) and os.path.exists(s12):
        Kref = read_K_from_out(ref)
        Ks12 = np.loadtxt(s12)
        dref = np.diag(Kref); ds12 = np.diag(Ks12)
        print("%-6s %14s %14s %9s" % ("term", "s12(0.2400)", "r0247(0.24696)", "%diff"))
        for j in range(6):
            pd = 100.0 * (ds12[j] - dref[j]) / dref[j]
            print("%-6s %14.5e %14.5e %8.2f%%" % (LBL[j], ds12[j], dref[j], pd))
    else:
        print("  missing:", "ref" if not os.path.exists(ref) else "", "s12" if not os.path.exists(s12) else "")


if __name__ == "__main__":
    main()
