#!/usr/bin/env bash
cd ~/OpenSG-TW-claude
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
PY=$HOME/miniconda3/envs/opensg_2_0/bin/python
$PY -u examples/TW-paper/xsec_paper/make_plots.py 2>&1 | tail -3
DST=~/claude_tmp/xsec_paper
cp examples/TW-paper/xsec_paper/figures/*.png $DST/figures/ 2>/dev/null
echo "=== figures in project ==="; ls $DST/figures/
cd $DST
which pdflatex >/dev/null 2>&1 || export PATH=$PATH:/usr/local/texlive/2024/bin/x86_64-linux:/opt/texlive/2023/bin/x86_64-linux
echo "=== pdflatex: $(which pdflatex 2>/dev/null || echo NOT-FOUND) ==="
pdflatex -interaction=nonstopmode main.tex > c1.log 2>&1
pdflatex -interaction=nonstopmode main.tex > c2.log 2>&1
echo "=== fatal errors ==="; grep -n '^!' c2.log | head -20
echo "=== output ==="; grep -E 'Output written' c2.log
ls -la main.pdf 2>/dev/null
