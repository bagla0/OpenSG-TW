#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
timeout 1200 $PY -u examples/taper_jax/2_hybrid_tube_4way.py > /tmp/hyb4way.log 2>&1
echo "RC=$?"
grep -E '================|diagonal:|max \|err\||wrote|Error|Traceback' /tmp/hyb4way.log | head -40
