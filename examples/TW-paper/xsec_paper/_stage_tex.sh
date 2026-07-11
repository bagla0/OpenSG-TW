#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
$PY -u examples/TW-paper/xsec_paper/emit_tables.py 2>&1 | tail -5
DST=~/claude_tmp/xsec_paper
mkdir -p $DST/tab
cp examples/TW-paper/xsec_paper/results/tex/*.tex $DST/tab/
cp examples/TW-paper/xsec_paper/figures/iea_r020_solid_prevabs.png \
   examples/TW-paper/xsec_paper/figures/iea_r030_solid_prevabs.png \
   examples/TW-paper/xsec_paper/figures/iea_r020_shell.png \
   examples/TW-paper/xsec_paper/figures/iea_r030_shell.png \
   examples/TW-paper/xsec_paper/figures/full_blade_rm_span.png \
   examples/TW-paper/xsec_paper/figures/conv_single_tube.png \
   examples/TW-paper/xsec_paper/figures/tube_thick_sweep.png \
   examples/TW-paper/xsec_paper/figures/single_tube_shell.png \
   examples/TW-paper/xsec_paper/figures/two_cell_shell.png $DST/figures/ 2>/dev/null
echo "=== staged tab/ ==="; ls $DST/tab/
echo "=== figures ==="; ls $DST/figures/ | grep -E 'prevabs|span|shell'
