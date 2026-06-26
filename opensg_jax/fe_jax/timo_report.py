"""Timoshenko 6x6 reporting helpers -- ALWAYS the full 6x6 matrix, benchmarked on EVERY non-zero term
(C11, C12, ... C66), never the diagonal alone.

Order is the OpenSG/VABS convention  [EA, GA2, GA3, GJ, EI2, EI3]  =  [ext, shear2, shear3, twist, bend2, bend3]
so C11=EA, C22=GA2, C33=GA3, C44=GJ, C55=EI2, C66=EI3 and the off-diagonals Cij are the couplings.

A term is treated as a real (non-zero) entry when |S_ij| >= max(|S|)/`neglect` (default 1000), i.e. within
three orders of magnitude of the largest stiffness; everything below that is structural zero and skipped.
"""
import numpy as np

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def sym(M):
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)


def print_6x6(C, name=""):
    """Print the full symmetric 6x6 Timoshenko matrix (order [EA, GA2, GA3, GJ, EI2, EI3])."""
    C = sym(C)
    head = ("%s " % name).lstrip()
    print("%sTimoshenko 6x6  [EA, GA2, GA3, GJ, EI2, EI3]:" % head)
    for i in range(6):
        print("  " + " ".join("%13.5e" % C[i, j] for j in range(6)))


def full_pcterr(C, S, neglect=1000.0):
    """6x6 %-error matrix; entries with |S_ij| below max/neglect are set to 0 (structural zeros)."""
    C, S = sym(C), sym(S)
    thr = np.max(np.abs(S)) / neglect
    out = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            out[i, j] = 100.0 * (C[i, j] - S[i, j]) / S[i, j] if abs(S[i, j]) >= thr else 0.0
    return out


def nonzero_terms(S, neglect=1000.0):
    """Upper-triangle (i<=j) indices of the non-zero terms of the reference S, with Cij labels."""
    S = sym(S)
    thr = np.max(np.abs(S)) / neglect
    out = []
    for i in range(6):
        for j in range(i, 6):
            if abs(S[i, j]) >= thr:
                tag = "C%d%d" % (i + 1, j + 1)
                if i == j:
                    tag += "(%s)" % LBL[i]
                out.append((i, j, tag))
    return out


def term_table(name, C, S, ref="benchmark", neglect=1000.0):
    """Full per-non-zero-term table (Cij notation) of model C vs reference S. Returns the worst |%err|.
    Drop-in for the old diagonal-only diag_table(name, C, S) but covers couplings too."""
    C, S = sym(C), sym(S)
    print("  %-11s %15s %15s %10s" % ("term", name, ref, "%err"))
    worst = 0.0
    for i, j, tag in nonzero_terms(S, neglect):
        e = 100.0 * (C[i, j] - S[i, j]) / S[i, j]
        print("  %-11s %15.5e %15.5e %+9.2f" % (tag, C[i, j], S[i, j], e))
        worst = max(worst, abs(e))
    print("  -> worst non-zero term: %.2f %%" % worst)
    return worst


def compare_terms(S, models, neglect=1000.0):
    """Per-non-zero-term comparison of several models against reference S (e.g. RM and KL vs 2-D solid).
    `models` = dict name->6x6. Prints one row per Cij term: reference value then each model's value and %diff."""
    S = sym(S)
    names = list(models)
    Ms = {k: sym(v) for k, v in models.items()}
    hdr = "  %-11s %14s" % ("term", "solid/ref")
    for k in names:
        hdr += " %14s %9s" % (k, k + "%d")
    print(hdr)
    worst = {k: 0.0 for k in names}
    for i, j, tag in nonzero_terms(S, neglect):
        row = "  %-11s %14.4e" % (tag, S[i, j])
        for k in names:
            e = 100.0 * (Ms[k][i, j] - S[i, j]) / S[i, j]
            row += " %14.4e %+8.2f" % (Ms[k][i, j], e)
            worst[k] = max(worst[k], abs(e))
        print(row)
    print("  -> worst non-zero term: " + ", ".join("%s %.2f%%" % (k, worst[k]) for k in names))
    return worst


def diag_table(name, C, S):
    """Back-compat diagonal-only table (kept so old cells still run); prefer term_table / compare_terms."""
    C, S = sym(C), sym(S)
    print("  %-5s %15s %15s %12s" % ("term", name, "benchmark", "%err"))
    for i in range(6):
        print("  %-5s %15.5e %15.5e %+11.4f" % (LBL[i], C[i, i], S[i, i], 100.0 * (C[i, i] - S[i, i]) / S[i, i]))
