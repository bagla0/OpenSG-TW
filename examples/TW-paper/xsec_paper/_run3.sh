#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
echo "===== PLOTS ====="
$PY -u examples/TW-paper/xsec_paper/make_plots.py 2>&1 | tail -6
echo "===== RECONCILE ====="
timeout 900 $PY -u examples/TW-paper/xsec_paper/_reconcile.py 2>&1 | tail -12
