"""collect_freq.py -- Nayak Example 4: the comparison TABLE (no plots) from
(a) the paper's Table 5 LITERATURE data -- Crawley's experiment, Crawley's
FEM, and Nayak's Reddy-HSDT 9-node FE on the 8x4 mesh (his best column) --
and (b) the ONE new result: the OpenSG-RM frequencies computed by ABAQUS
*FREQUENCY on the MSG-ABDG general-section S4 model (make_abaqus_freq.py),
parsed from Abaqus_results/ex4_<slug>_freq.dat.

Output: ex4_freq_table.dat with TWO percent-error columns, both taken
against Crawley's FEM [18]: Nayak's Reddy FE vs FEM, and OpenSG-RM vs FEM.

Run:  python examples/opensg-rm_dynamic/ex4/collect_freq.py
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from make_abaqus_freq import LAYUPS, SLUGS, TABLE5, NMODES


def read_eigenfrequencies(dat_path):
    """The CYCLES/TIME column of the Abaqus .dat eigenvalue table."""
    freqs = []
    active = False
    with open(dat_path, errors="replace") as f:
        for ln in f:
            if "E I G E N V A L U E    O U T P U T" in ln:
                active = True
                continue
            toks = ln.split()
            if active and toks and re.fullmatch(r"\d+", toks[0]):
                # MODE NO, EIGENVALUE, FREQ(RAD/TIME), FREQ(CYCLES/TIME), ..
                freqs.append(float(toks[3]))
            elif active and freqs and toks and not \
                    re.fullmatch(r"\d+", toks[0]):
                break
    return np.array(freqs)


def main():
    lines = ["Nayak Ex.4 (Crawley cantilever sandwich): frequencies [Hz]",
             "OpenSG-RM = Abaqus S4 *FREQUENCY with the MSG ABDG general"
             " section (make_abaqus_freq.py); Reddy = Nayak's 9-node HSDT"
             " FE, 8x4 mesh (Table 5); %err columns vs Crawley's FEM [18]",
             "%-18s %-6s %10s %12s %12s %12s %14s %14s" %
             ("layup", "mode", "Expt[18]", "FEM[18]", "Reddy-8x4",
              "OpenSG-RM", "%err(Reddy-FEM)", "%err(RM-FEM)")]
    for name in LAYUPS:
        slug = SLUGS[name]
        dat = os.path.join(HERE, "Abaqus_results", "ex4_%s_freq.dat" % slug)
        if not os.path.isfile(dat):
            print("missing %s -- run the Abaqus job first" % dat)
            continue
        f_rm = read_eigenfrequencies(dat)[:NMODES]
        expt, fem, p9 = TABLE5[name]
        for m in range(NMODES):
            lines.append("%-18s %-6d %10.1f %12.1f %12.1f %12.1f %+14.2f"
                         " %+14.2f" %
                         (name if m == 0 else "", m + 1, expt[m], fem[m],
                          p9[m], f_rm[m],
                          100 * (p9[m] - fem[m]) / fem[m],
                          100 * (f_rm[m] - fem[m]) / fem[m]))
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(HERE, "ex4_freq_table.dat"), "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
