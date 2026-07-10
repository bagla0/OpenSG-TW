#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
timeout 3000 $PY -u examples/taper_jax/6_thin_02h_study.py > /tmp/thin02h.log 2>&1
echo "RC=$?"
grep -E 'THIN-WALL study|################|solid mixed YAML|JAX solid|RM shell|wrote|INVERTED|segment diag|ALL SEGMENTS|Error|Traceback' /tmp/thin02h.log | head -30
