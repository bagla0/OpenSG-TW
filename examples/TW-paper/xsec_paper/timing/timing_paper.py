"""TASK A harness -- genuine sequential wall-clock timings: 6 tube cases x 3 models.

Each (case, model) runs in a FRESH python subprocess, strictly one at a time:
  RM / KL  -> worker_shell.py under the opensg_2_0 JAX env
  solid    -> worker_solid.py under the opensg_env_v8 dolfinx env
wall_s = time.perf_counter around subprocess.run  (fresh-process end-to-end:
interpreter + imports + mesh/ABD build + solve, JAX JIT included).  The worker
additionally reports t_import, t_compute (cold call only, JIT included) and
t_warm (2nd call in the same process, JIT cached; shell only).

Writes timing_taskA.json and streams timing_taskA.log.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.expanduser("~/OpenSG-TW-claude/examples/TW-paper/reproduce_6x6")
PY20 = os.path.expanduser("~/miniconda3/envs/opensg_2_0/bin/python")
PYV8 = os.path.expanduser("~/miniconda3/envs/opensg_env_v8/bin/python")

CASES = ["single_rh02", "single_rh10",
         "2cell_iso_thin", "2cell_iso_thick",
         "2cell_aniso_thin", "2cell_aniso_thick"]
SOLID_YAML = {
    "single_rh02": "solid_rh02.yaml",
    "single_rh10": "solid_rh10.yaml",
    "2cell_iso_thin": "solid_tube2cell_thin.yaml",
    "2cell_iso_thick": "solid_tube2cell_thick.yaml",
    "2cell_aniso_thin": "solid_tube2cell_aniso_thin.yaml",
    "2cell_aniso_thick": "solid_tube2cell_aniso_thick.yaml",
}

LOG = open(os.path.join(HERE, "timing_taskA.log"), "w")


def say(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def run_one(cmd):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    rec = None
    for line in r.stdout.splitlines():
        if line.startswith("###JSON### "):
            rec = json.loads(line[len("###JSON### "):])
    if rec is None:
        rec = dict(case=cmd[-2], model="?", error=(r.stderr or r.stdout)[-2000:])
    rec["wall_s"] = wall
    rec["returncode"] = r.returncode
    return rec


results = []
say("TASK A start %s  strictly sequential;  loadavg %s"
    % (time.strftime("%Y-%m-%d %H:%M:%S"), open("/proc/loadavg").read().strip()))
for case in CASES:
    for model in ("RM", "KL"):
        say("-> %s %s" % (case, model))
        rec = run_one([PY20, os.path.join(HERE, "worker_shell.py"), case, model])
        results.append(rec)
        say("   %s" % json.dumps(rec))
    say("-> %s solid" % case)
    rec = run_one([PYV8, os.path.join(HERE, "worker_solid.py"), case,
                   os.path.join(R6, "reference", SOLID_YAML[case])])
    results.append(rec)
    say("   %s" % json.dumps(rec))

out = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"),
           loadavg_end=open("/proc/loadavg").read().strip(),
           results=results)
with open(os.path.join(HERE, "timing_taskA.json"), "w") as f:
    json.dump(out, f, indent=1)
say("TASK A done %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
