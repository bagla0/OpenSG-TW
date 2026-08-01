"""collect_freq.py -- Nayak Example 4: build the comparison table + figures
from (a) the paper's Table 5 LITERATURE data -- Crawley's experiment,
Crawley's FEM, and Nayak's Reddy-HSDT 9-node FE (his converged 8x4 mesh
column) -- and (b) the ONE new result: the OpenSG-RM frequencies computed by
ABAQUS *FREQUENCY on the MSG-ABDG general-section S4 model
(make_abaqus_freq.py), parsed from Abaqus_results/ex4_<slug>_freq.dat.

Outputs: ex4_freq_table.dat + one figure per layup (each with its own
legend): Experiment (dashed black), Crawley FEM (gray), Nayak Reddy HSDT
(steel blue), OpenSG-RM/Abaqus (orange markers).

Run:  python examples/opensg-rm_dynamic/ex4/collect_freq.py
"""
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
             " section (make_abaqus_freq.py); the rest = Nayak Table 5",
             "%-18s %-6s %10s %12s %14s %14s %8s" %
             ("layup", "mode", "Expt[18]", "CrawleyFEM", "Nayak-Reddy-P9",
              "OpenSG-RM-Abq", "%vsExpt")]
    for name in LAYUPS:
        slug = SLUGS[name]
        dat = os.path.join(HERE, "Abaqus_results", "ex4_%s_freq.dat" % slug)
        if not os.path.isfile(dat):
            print("missing %s -- run the Abaqus job first" % dat)
            continue
        f_rm = read_eigenfrequencies(dat)[:NMODES]
        expt, fem, p9 = TABLE5[name]
        for m in range(NMODES):
            lines.append("%-18s %-6d %10.1f %12.1f %14.1f %14.1f %+8.1f" %
                         (name if m == 0 else "", m + 1, expt[m], fem[m],
                          p9[m], f_rm[m],
                          100 * (f_rm[m] - expt[m]) / expt[m]))
        modes = np.arange(1, NMODES + 1)
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.plot(modes, expt, "--o", color="k", lw=1.6, ms=6, mfc="none",
                mew=1.3, label="Experiment (Crawley)")
        ax.plot(modes, fem, "-d", color="0.55", lw=1.3, ms=5, mfc="none",
                mew=1.1, label="Crawley FEM")
        ax.plot(modes, p9, "-^", color="#4878a8", lw=1.4, ms=6, mfc="none",
                mew=1.2, label="Nayak Reddy-HSDT FE\n(9-node, 8x4)")
        ax.plot(modes, f_rm, "-s", color="#ff7f0e", lw=1.4, ms=6,
                mfc="none", mew=1.2, label="OpenSG-RM\n(Abaqus S4 + MSG ABDG)")
        ax.set_xlabel("mode", fontsize=11)
        ax.set_ylabel("frequency [Hz]", fontsize=11)
        ax.set_xticks(modes)
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "ex4_freq_%s.png" % slug), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(HERE, "ex4_freq_table.dat"), "w") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
