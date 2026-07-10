#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
timeout 200 $PY -u examples/taper_jax/0_gen_ellipse_tet_segment.py 2>&1 | tail -2
timeout 900 $PY -u examples/taper_jax/1_run_solid_taper_jax.py examples/taper_jax/ellipse_tet_seg.yaml --fenics > /tmp/elltet.log 2>&1
echo "RC=$?"
grep -E 'batches|elements=|max \|err\||diagonal: EA|Error|Traceback' /tmp/elltet.log | head -12
