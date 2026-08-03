"""Step 4 - percent-error of KL and RM vs the FEniCS-2D-solid (VABS) reference,
and a side-by-side check against the numbers printed in the ASC-2026 paper tables.

%error(term) = 100 * (shell - solid) / solid, per Timoshenko term.
Prints one block per case and writes results/RESULTS_verification.txt.

Term map:  C11=EA  C22=GA2  C33=GA3  C44=GJ  C55=EI2  C66=EI3 ;
           C14=ext-twist  C25=GA2-EI2  C36=GA3-EI3.
"""
import os

from common import cases, load6, pe, RES, TERM_IJ

# Paper-table percent errors  name -> {term: (KL%, RM%)}
PAPER = {
    "single_rh02": {"C11": (-0.39, -0.39), "C14": (-5.38, -5.38), "C22": (-37.70, -7.98),
                    "C25": (-35.48, -4.92), "C33": (-38.07, -7.98), "C36": (-35.90, -4.92),
                    "C44": (6.58, 7.16), "C55": (-8.74, -4.99), "C66": (-8.80, -4.99)},
    "single_rh10": {"C11": (-0.01, -0.01), "C14": (-0.21, -0.21), "C22": (-2.47, -0.43),
                    "C25": (-2.27, -0.23), "C33": (-2.46, -0.43), "C36": (-2.25, -0.23),
                    "C44": (0.33, 0.33), "C55": (-0.45, -0.20), "C66": (-0.45, -0.20)},
    "2cell_iso_thin":   {"C11": (1.0, 1.0), "C22": (-4.8, -1.1), "C33": (-4.5, -0.2),
                         "C44": (-0.3, -0.3), "C55": (2.0, 2.0), "C66": (-0.1, -0.1)},
    "2cell_iso_thick":  {"C11": (4.2, 4.2), "C22": (-40.0, -6.8), "C33": (-38.9, -3.2),
                         "C44": (0.4, 0.3), "C55": (6.3, 6.2), "C66": (-1.4, -1.6)},
    "2cell_aniso_thin": {"C11": (0.9, 1.0), "C22": (-14.3, -2.0), "C33": (-11.2, -0.2),
                         "C44": (-0.3, -0.3), "C55": (0.2, 1.5), "C66": (-0.9, -0.2),
                         "C14": (-0.3, -0.3)},
    "2cell_aniso_thick":{"C11": (3.6, 4.0), "C22": (-71.9, -7.6), "C33": (-63.0, -3.3),
                         "C44": (-1.3, -0.2), "C55": (-4.2, 4.3), "C66": (-6.5, -2.2),
                         "C14": (-4.2, -3.6)},
}

_lines = []


def out(s=""):
    print(s)
    _lines.append(s)


out("=" * 92)
out("VERIFICATION  --  rerun KL/RM Timoshenko 6x6  vs  FEniCS-2D-solid  vs  ASC-2026 paper table")
out("%error = 100*(shell-solid)/solid.  C11=EA C22=GA2 C33=GA3 C44=GJ C55=EI2 C66=EI3.")
out("=" * 92)

worst = 0.0
for c in cases():
    kl_p = os.path.join(RES, "C6_KL_%s.dat" % c["name"])
    rm_p = os.path.join(RES, "C6_RM_%s.dat" % c["name"])
    if not (os.path.exists(kl_p) and os.path.exists(rm_p) and os.path.exists(c["solid"])):
        out("\n--- %-18s  SKIP (missing results or solid reference) ---" % c["name"])
        continue
    KL, RM, S = load6(kl_p), load6(rm_p), load6(c["solid"])
    paper = PAPER.get(c["name"])
    out("\n--- %-18s  (solid EA=%.4e) ---" % (c["name"], S[0, 0]))
    if paper:
        out("  term     solid Cij       KL%   (paper)   dKL      RM%   (paper)   dRM")
        cmax = 0.0
        for term in paper:
            i, j = TERM_IJ[term]
            kl, rm = pe(KL[i, j], S[i, j]), pe(RM[i, j], S[i, j])
            pk, pr = paper[term]
            dk, dr = kl - pk, rm - pr
            cmax = max(cmax, abs(dk), abs(dr))
            out("  %-4s %13.4e   %+7.2f (%+6.2f) %+6.2f   %+7.2f (%+6.2f) %+6.2f"
                % (term, S[i, j], kl, pk, dk, rm, pr, dr))
        out("  -> max |delta vs paper| = %.2f percentage points" % cmax)
        worst = max(worst, cmax)
    else:
        out("  term     solid Cij       KL%       RM%   (no table row -- sweep-figure data)")
        for term in ("C11", "C22", "C33", "C44", "C55", "C66"):
            i, j = TERM_IJ[term]
            out("  %-4s %13.4e   %+7.2f   %+7.2f"
                % (term, S[i, j], pe(KL[i, j], S[i, j]), pe(RM[i, j], S[i, j])))

out("\n" + "=" * 92)
out("OVERALL max |delta vs paper table| across all tabulated cases = %.2f percentage points" % worst)
out("=" * 92)

path = os.path.join(RES, "RESULTS_verification.txt")
open(path, "w").write("\n".join(_lines) + "\n")
print("\nwrote %s" % path)
