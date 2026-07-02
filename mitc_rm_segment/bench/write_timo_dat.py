"""
write_timo_dat.py    [ Windows opensg_2_0_env ]
========================================================================
Tapered BAR-URC blade: JAX MITC-RM shell Timoshenko 6x6 vs FEniCS SOLID
(OpenSG-1.0 compute_stiffness, Taper=True), SAME segment / SAME origin,
segment by segment.  Writes bench/out/timo_segment_<id>.dat with the full 6x6
(all non-zero terms) for shell + solid and the per-term % error.

Prereq numbers cached by:
  * shell : compute_timo_taper(..., return_timo=True)          (Windows)
  * solid : run_solid_segment.py  (WSL opensg_env_v8)          -> solid_seg<id>_6x6.npy
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEG = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, SEG)
from compute_timo_taper import compute_timo_taper

CACHE = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\bagla0\AppData\Local\Temp\claude\C--Users-bagla0\91cf4f05-ed42-47e2-974c-813d98a91247\scratchpad"
OUTDIR = os.path.join(HERE, "out"); os.makedirs(OUTDIR, exist_ok=True)
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
NAME = {(0, 0): "C11(EA)", (1, 1): "C22(GA2)", (2, 2): "C33(GA3)", (3, 3): "C44(GJ)",
        (4, 4): "C55(EI2)", (5, 5): "C66(EI3)"}


def mat_str(M):
    return "\n".join("  " + "".join("%15.5e" % M[i, j] for j in range(6)) for i in range(6))


def do_segment(segid, full_curvature=False):
    sh = os.path.join(CACHE, "solid_seg%d_6x6.npy" % segid)
    if not os.path.exists(sh):
        print("  solid segment %d not ready (%s missing) -- skipping" % (segid, os.path.basename(sh)))
        return
    solid = np.load(sh)
    b = np.load(os.path.join(SEG, "out", "BAR_URC_numEl_52_segment_%d.npz" % segid), allow_pickle=True)
    r = compute_timo_taper(b, k22_mode="general", return_timo=True, full_curvature=full_curvature)
    shell = np.asarray(r["C6"]); sh_org = r["origin"]
    so_org = float(open(os.path.join(CACHE, "solid_seg%d_origin.txt" % segid)).read().strip()) \
        if os.path.exists(os.path.join(CACHE, "solid_seg%d_origin.txt" % segid)) else float("nan")

    thr = np.max(np.abs(solid)) / 1000.0                       # non-zero-term cutoff
    lines = []
    lines.append("# BAR-URC tapered segment %d  (JAX MITC-RM shell vs FEniCS solid)" % segid)
    lines.append("# origin: shell=%.4f  solid=%.4f   full_curvature=%s" % (sh_org, so_org, full_curvature))
    lines.append("# Timoshenko 6x6 order [EA, GA2, GA3, GJ, EI2, EI3]")
    lines.append("#\n# --- SHELL (JAX MITC-RM) 6x6 ---"); lines.append(mat_str(shell))
    lines.append("#\n# --- SOLID (FEniCS) 6x6 ---"); lines.append(mat_str(solid))
    lines.append("#\n# --- per-term %% error (shell vs solid), |term| > max/1000 ---")
    lines.append("# %-12s %15s %15s %10s" % ("term", "shell", "solid", "%err"))
    print("\n=== segment %d  (origin shell %.3f / solid %.3f) ===" % (segid, sh_org, so_org))
    print("%-12s %15s %15s %10s" % ("term", "shell", "solid", "%err"))
    for i in range(6):
        for j in range(i, 6):
            if abs(solid[i, j]) <= thr:
                continue
            e = 100.0 * (shell[i, j] - solid[i, j]) / solid[i, j]
            nm = NAME.get((i, j), "C%d%d" % (i + 1, j + 1))
            lines.append("# %-12s %15.5e %15.5e %+9.2f%%" % (nm, shell[i, j], solid[i, j], e))
            print("%-12s %15.5e %15.5e %+9.2f%%" % (nm, shell[i, j], solid[i, j], e))
    fn = os.path.join(OUTDIR, "timo_segment_%d.dat" % segid)
    open(fn, "w").write("\n".join(lines) + "\n")
    print("  wrote", fn)


if __name__ == "__main__":
    for sid in (0, 5):
        do_segment(sid)
