"""Merge timing_taskA.json + timing_taskB.json -> timing_results.json + timing_results.log.

Also cross-checks every recomputed 6x6 diagonal against the shipped reproduce_6x6
numbers (results/C6_RM_*.dat, results/C6_KL_*.dat, reference/C6_solid_*.txt) and
records the max |%diff| over the 6 diagonal terms per run (sanity that the timed
run really computed the paper's numbers).
"""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.expanduser("~/OpenSG-TW-claude/examples/TW-paper/reproduce_6x6")

SOLID_TXT = {
    "single_rh02": "C6_solid_rh02.txt",
    "single_rh10": "C6_solid_rh10.txt",
    "2cell_iso_thin": "C6_solid_tube2cell_thin.txt",
    "2cell_iso_thick": "C6_solid_tube2cell_thick.txt",
    "2cell_aniso_thin": "C6_solid_tube2cell_aniso_thin.txt",
    "2cell_aniso_thick": "C6_solid_tube2cell_aniso_thick.txt",
}


def cpu_model():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def ref_diag(case, model):
    try:
        if model == "RM":
            p = os.path.join(R6, "results", "C6_RM_%s.dat" % case)
        elif model == "KL":
            p = os.path.join(R6, "results", "C6_KL_%s.dat" % case)
        else:
            p = os.path.join(R6, "reference", SOLID_TXT[case])
        M = np.loadtxt(p)
        return np.diag(M)
    except Exception:
        return None


A = json.load(open(os.path.join(HERE, "timing_taskA.json")))
B = json.load(open(os.path.join(HERE, "timing_taskB.json")))

for rec in A["results"]:
    rd = ref_diag(rec.get("case", ""), rec.get("model", ""))
    if rd is not None and "diag" in rec:
        d = np.asarray(rec["diag"])
        rec["check_max_pct_vs_shipped"] = float(np.max(np.abs(100.0 * (d - rd) / rd)))

machine = "%s, %d cores, %s" % (cpu_model(), os.cpu_count(), os.uname().nodename)

out = dict(
    machine=machine,
    generated=time.strftime("%Y-%m-%d %H:%M:%S"),
    protocol=dict(
        taskA="each (case, model) in a FRESH python subprocess, strictly sequential; "
              "wall_s = perf_counter around subprocess.run (interpreter+imports+solve, JIT incl.); "
              "t_compute = cold in-process entry-point call (JIT incl.); t_warm = 2nd call (JIT cached)",
        taskB="one driver process, 51 stations strictly sequential, per-station perf_counter; "
              "entry point ring_6dof(load_ring(yaml)) as in homo_rm_shell.py, center-ref shell51/1d_yaml"),
    taskA=A, taskB=B)

with open(os.path.join(HERE, "timing_results.json"), "w") as f:
    json.dump(out, f, indent=1)

L = []
L.append("GENUINE WALL-CLOCK TIMINGS  --  %s" % out["generated"])
L.append("machine: %s" % machine)
L.append("")
L.append("TASK A: composite-tube Timoshenko 6x6 homogenization (fresh process per run, sequential)")
L.append("%-20s %-6s %8s %10s %10s %10s %12s" % ("case", "model", "n_elem",
         "wall_s", "compute_s", "warm_s", "chk%vs_ship"))
for r in A["results"]:
    L.append("%-20s %-6s %8s %10.2f %10.2f %10s %12s" % (
        r.get("case", "?"), r.get("model", "?"), str(r.get("n_elem", "-")),
        r.get("wall_s", float("nan")), r.get("t_compute", float("nan")),
        ("%.2f" % r["t_warm"]) if r.get("t_warm") else "-",
        ("%.3f" % r["check_max_pct_vs_shipped"]) if "check_max_pct_vs_shipped" in r else "-"))
L.append("")
L.append("TASK B: IEA-22 51-station RM homogenization (center-ref, one process, sequential)")
L.append("stations ok/found: %d/%d   total %.1f s   avg %.2f s/station   (first station includes JIT)"
         % (B["n_ok"], B["n_files"], B["total_s"], B["avg_s"]))
st = [p["wall_s"] for p in B["per_station"]]
if st:
    L.append("per-station wall_s: first=%.2f  min=%.2f  median=%.2f  max=%.2f"
             % (st[0], min(st), sorted(st)[len(st) // 2], max(st)))
for fl in B.get("failures", []):
    L.append("FAIL %s: %s" % (fl["station"], fl["error"]))
with open(os.path.join(HERE, "timing_results.log"), "w") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
