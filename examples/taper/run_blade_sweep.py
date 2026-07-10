"""run_blade_sweep.py -- whole-blade tapered-segment sweep, shell vs solid.

Marches the IEA-22 blade root->tip in segments; for EACH segment it runs the full
numbered pipeline (1-6) via subprocess (so every step is exactly the standalone,
reproducible script), collects the Timoshenko 6x6 of the shell and solid taper AND both
boundary rings, writes a per-segment comparison .dat with the %-error 6x6, and stages a
SEPARATE per-segment folder (meshes + results + logs) into the OneDrive deliverable.
A master summary tracks the diagonal shell-vs-solid %-error along the span.

Run (server opensg_2_0):
    python examples/taper/run_blade_sweep.py            # default segments 0.05..0.95
    python examples/taper/run_blade_sweep.py 0.2 0.3    # a single segment
"""
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
ONEDRIVE = os.environ.get("SWEEP_OUT", os.path.join(HERE, "sweep_out"))
CUT = 1.0e6
# default: 0.1-wide segments spanning the airfoil range (root cylinder / tip excluded)
DEFAULT = [(round(0.05 + 0.1 * k, 2), round(0.15 + 0.1 * k, 2)) for k in range(9)]


def run(step, *args, log=None):
    cmd = [PY, os.path.join(HERE, step)] + [str(a) for a in args]
    print("  $", os.path.basename(step), *[str(a) for a in args], flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if log:
        open(log, "a").write("\n\n########## %s %s\n%s" % (step, args, out))
    if r.returncode != 0:
        print(out[-2000:]); raise RuntimeError("%s failed" % step)
    return out


def fmt66(f, name, M):
    f.write("\n-- %s --\n" % name)
    f.write("        " + "".join("%13s" % c for c in LBL) + "\n")
    for i in range(6):
        f.write("  %-4s" % LBL[i] + "".join("%13.4e" % M[i, j] for j in range(6)) + "\n")


def err66(f, H, S):
    f.write("\n-- %%ERROR 6x6 (shell vs solid) --\n")
    f.write("        " + "".join("%13s" % c for c in LBL) + "\n")
    for i in range(6):
        row = "  %-4s" % LBL[i]
        for j in range(6):
            row += ("%12.2f%%" % (100 * (H[i, j] - S[i, j]) / S[i, j])) if abs(S[i, j]) >= CUT else "%13s" % "."
        f.write(row + "\n")


def comparison(seg_dir, name, r1, r2):
    ss = np.load(os.path.join(seg_dir, "solid_segment_timo.npz"))
    sh = np.load(os.path.join(seg_dir, "shell_segment_timo.npz"))
    blocks = [("TAPER SEGMENT r=%.2f->%.2f (3-D)" % (r1, r2), sh["S6"], ss["S6"])]
    # boundary rings: prefer the standalone boundary solves (scripts 3/5) if present,
    # else fall back to the end rings the segment solvers already returned (C6L/C6R)
    for side in ("L", "R"):
        try:
            bs = np.load(os.path.join(seg_dir, "solid_boundary_%s_timo.npz" % side))["C6"]
            bh = np.load(os.path.join(seg_dir, "shell_boundary_%s_timo.npz" % side))["C6"]
        except Exception:
            bs, bh = ss["C6" + side], sh["C6" + side]
        blocks.append(("BOUNDARY %s ring (r=%.2f)" % (side, r1 if side == "L" else r2), bh, bs))
    dat = os.path.join(seg_dir, "comparison.dat")
    with open(dat, "w") as f:
        f.write("# IEA-22 blade tapered segment %s : SHELL (TW JAX RM) vs SOLID (FEniCS)\n" % name)
        f.write("# Timoshenko 6x6 [EA,GA2,GA3,GJ,EI2,EI3]; %%err='.' where |solid|<%.0e\n" % CUT)
        f.write("# solid DOF %d (%.1fs) | shell DOF %d (%.1fs)\n"
                % (ss["dof"], ss["time"], sh["dof"], sh["time"]))
        for nm, H, S in blocks:
            f.write("\n" + "=" * 74 + "\n### %s\n" % nm + "=" * 74 + "\n")
            fmt66(f, "SOLID", S); fmt66(f, "SHELL", H); err66(f, H, S)
            f.write("  diag %%err: " + "  ".join("%s %+.1f%%" % (LBL[i], 100 * (H[i, i] - S[i, i]) / S[i, i])
                                                 for i in range(6)) + "\n")
    return sh["S6"], ss["S6"]


def main(segments):
    os.makedirs(ONEDRIVE, exist_ok=True)
    shutil.copytree(HERE, os.path.join(ONEDRIVE, "scripts"),
                    ignore=shutil.ignore_patterns("out*", "_*", "__pycache__"), dirs_exist_ok=True)
    summary = []
    for (r1, r2) in segments:
        name = "r%03d_%03d" % (round(r1 * 100), round(r2 * 100))
        print("\n==================== SEGMENT %s ====================" % name, flush=True)
        seg = os.path.join(ONEDRIVE, name); os.makedirs(seg, exist_ok=True)
        log = os.path.join(seg, "run.log"); open(log, "w").close()
        try:
            run("1_generate_solid_mesh.py", r1, r2, seg, log=log)
            run("2_generate_shell_mesh.py", r1, r2, seg, log=log)
            # standalone boundary solves (scripts 3/5) on the first segment only (they are
            # slow and the segment solves already return the end rings); set FULL_BOUNDARY=1
            # to run them on every segment.
            if os.environ.get("FULL_BOUNDARY") or not summary:
                for side in ("L", "R"):
                    run("3_get_beam_props_from_solid_boundary.py", os.path.join(seg, "solid_boundary_%s.yaml" % side), log=log)
                    run("5_get_beam_props_from_shell_boundary.py", os.path.join(seg, "shell_boundary_%s.yaml" % side), log=log)
            run("6_get_beam_props_from_shell_segment.py", os.path.join(seg, "shell_segment.yaml"), log=log)
            run("4_get_beam_props_from_solid_segment.py", os.path.join(seg, "solid_segment.yaml"), log=log)
            H, S = comparison(seg, name, r1, r2)
            summary.append((r1, r2, H, S))
            print("  -> %s : diag %%err " % name +
                  "  ".join("%s%+.0f" % (LBL[i], 100 * (H[i, i] - S[i, i]) / S[i, i]) for i in range(6)),
                  flush=True)
        except Exception as e:
            print("  SEGMENT %s FAILED: %s" % (name, e), flush=True)
    # master span summary
    with open(os.path.join(ONEDRIVE, "SUMMARY_diag_error_vs_span.dat"), "w") as f:
        f.write("# IEA-22 blade shell-vs-solid Timoshenko DIAGONAL %-error along the span\n")
        f.write("# %-14s " % "segment" + "".join("%9s" % c for c in LBL) + "\n")
        for (r1, r2, H, S) in summary:
            f.write("  r=%.2f-%.2f    " % (r1, r2) +
                    "".join("%+8.1f%%" % (100 * (H[i, i] - S[i, i]) / S[i, i]) for i in range(6)) + "\n")
    print("\nSWEEP done ->", ONEDRIVE, flush=True)


if __name__ == "__main__":
    segs = [(float(sys.argv[1]), float(sys.argv[2]))] if len(sys.argv) > 2 else DEFAULT
    main(segs)
