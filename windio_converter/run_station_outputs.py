"""CANONICAL per-station OpenSG output pipeline -- enforces the 3 MANDATORY outputs for every run:

  1. orientation PNG for BOTH the 2D-solid AND the 1D-shell  -> orient_stations/orient_<case>_<tag>.png
  2. RM and KL Timoshenko 6x6 (bare .K convention)            -> K_matrices/<case>_<tag>_{RM,KL,solid}.K
  3. %-error 6x6 of RM and KL vs the 2D-solid (|term|<1e6->0)  -> <case>_pcterr_{RM,KL}.dat

Inputs per station: a 1D-shell YAML and a 2D-solid YAML (+ its solid Timo 6x6 C6_solid_<case>_<tag>.txt,
produced by the FEniCS solid solver on the server). Run AFTER the solids exist.

Usage:  python run_station_outputs.py [case] [tagA tagB ...]
        defaults: case=iea22, tags r010..r090 + r095
"""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff
from fe_jax.orient_plot import plot_orient

VAL = os.path.join(CC, "windio_converter", "validation")
KDIR = os.path.join(VAL, "K_matrices"); os.makedirs(KDIR, exist_ok=True)
ODIR = os.path.join(VAL, "orient_stations"); os.makedirs(ODIR, exist_ok=True)


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


def mat6(M):
    return "\n".join(" ".join("% .6e" % M[i, j] for j in range(6)) for i in range(6))


def pct6(M, S):
    # MANDATORY rule: %-error only where |term| >= 1e6; smaller terms are denominator-noise -> 0.
    out = []
    for i in range(6):
        row = ["% 9.2f" % (100 * (M[i, j] - S[i, j]) / S[i, j]) if abs(S[i, j]) >= 1.0e6 else "% 9.2f" % 0.0
               for j in range(6)]
        out.append(" ".join(row))
    return "\n".join(out)


def run_station(case, tag):
    """Do all 3 mandatory steps for one station. Returns (S, RM, KL) or None if shell YAML missing."""
    shell = os.path.join(VAL, "shell_%s_%s.yaml" % (case, tag))
    solid = os.path.join(VAL, "solid_%s_%s.yaml" % (case, tag))
    if not os.path.exists(shell):
        print("  [skip] no shell YAML for %s %s" % (case, tag), flush=True); return None
    # (1) orientation PNG -- solid + shell e1/e2/e3
    try:
        png = os.path.join(ODIR, "orient_%s_%s.png" % (case, tag))
        plot_orient(shell, solid if os.path.exists(solid) else None, png)
    except Exception as e:
        print("  [warn] orient PNG failed for %s: %s" % (tag, e), flush=True)
    # (2) RM + KL 6x6, bare .K
    RM = sym(rm_timoshenko_6x6(shell, 0.0, orient=False))
    KL = sym(gradient_junction_kirchhoff(shell, frac=0.0, orient=False)[0])
    np.savetxt(os.path.join(KDIR, "%s_%s_RM.K" % (case, tag)), RM, fmt="% .6e")
    np.savetxt(os.path.join(KDIR, "%s_%s_KL.K" % (case, tag)), KL, fmt="% .6e")
    np.savetxt(os.path.join(VAL, "C6_RM_%s_%s.txt" % (case, tag)), RM, fmt="% .6e")
    np.savetxt(os.path.join(VAL, "C6_KL_%s_%s.txt" % (case, tag)), KL, fmt="% .6e")
    sp = os.path.join(VAL, "C6_solid_%s_%s.txt" % (case, tag))
    S = sym(np.loadtxt(sp)) if os.path.exists(sp) else None
    if S is not None:
        np.savetxt(os.path.join(KDIR, "%s_%s_solid.K" % (case, tag)), S, fmt="% .6e")
    return (S, RM, KL)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "iea22"
    tags = sys.argv[2:] if len(sys.argv) > 2 else ["r%03d" % (10 * k) for k in range(1, 10)] + ["r095"]
    pr, pk = [], []
    for tag in tags:
        print("station %s %s" % (case, tag), flush=True)
        res = run_station(case, tag)
        if res is None:
            continue
        S, RM, KL = res
        if S is None:
            print("  [note] solid Timo missing (run FEniCS solid first) -- %%err skipped for %s" % tag, flush=True)
            continue
        # (3) %-error 6x6 (1e6 cutoff) for RM and KL
        pr.append("\n# %s\n%s\n" % (tag, pct6(RM, S)))
        pk.append("\n# %s\n%s\n" % (tag, pct6(KL, S)))
        d = lambda M: [100 * (M[i, i] - S[i, i]) / S[i, i] for i in range(6)]
        print("  RM diag%%: " + " ".join("%+6.1f" % x for x in d(RM)), flush=True)
        print("  KL diag%%: " + " ".join("%+6.1f" % x for x in d(KL)), flush=True)
    if pr:
        hdr = "# %s %%-error vs 2D-solid, full 6x6 (|term|<1e6 -> 0). Order: axial, shear-y, shear-z, twist, bend-y, bend-z\n"
        open(os.path.join(VAL, "%s_pcterr_RM.dat" % case), "w").write(hdr % case + "".join(pr))
        open(os.path.join(VAL, "%s_pcterr_KL.dat" % case), "w").write(hdr % case + "".join(pk))
        print("\nwrote orient_stations/*.png, K_matrices/*.K, %s_pcterr_{RM,KL}.dat -> validation/" % case, flush=True)


if __name__ == "__main__":
    main()
