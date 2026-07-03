"""Convergence overview for the taper study: per case (regime x mat), the
max-|%err| of the RM vs solid 6x6 diagonal for L boundary / taper / R boundary
as the taper rate grows.  Also writes summary_<regime>_<mat>.dat."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "out", "taper_study", "results")
TAPERS = [1.0, 0.95, 0.9, 0.8, 0.7]
LBL = ["C11", "C22", "C33", "C44", "C55", "C66"]


def one(regime, mat):
    print("\n===== %s %s : RM(general) vs solid, diagonal %%err =====" % (regime, mat))
    hdr = "%-6s | %s | %s | %s" % ("aR",
                                   " ".join("%6s" % k for k in LBL) + "  (LEFT)",
                                   " ".join("%6s" % k for k in LBL) + "  (TAPER)",
                                   " ".join("%6s" % k for k in LBL) + "  (RIGHT)")
    print(hdr)
    lines = [hdr]
    for aR in TAPERS:
        tg = "%s_%s_aR%03d" % (regime, mat, round(aR * 100))
        row = ["%-6.2f" % aR]
        for part in ("L", "seg", "R"):
            try:
                Sh = np.load(os.path.join(RES, "rm_%s_%s.npy" % (tg, part)))
                So = np.load(os.path.join(RES, "solid_%s_%s.npy" % (tg, part)))
            except FileNotFoundError:
                row.append("  (missing)")
                continue
            e = [100 * (Sh[i, i] - So[i, i]) / So[i, i] for i in range(6)]
            row.append(" ".join("%+5.1f%%" % v for v in e))
        line = " | ".join(row)
        print(line); lines.append(line)
    fn = os.path.join(RES, "summary_%s_%s.dat" % (regime, mat))
    open(fn, "w").write("\n".join(lines) + "\n")
    return fn


if __name__ == "__main__":
    which = sys.argv[1:] or ["thin_iso", "thick_iso", "thin_m45", "thick_m45"]
    for w in which:
        regime, mat = w.split("_")
        one(regime, mat)
