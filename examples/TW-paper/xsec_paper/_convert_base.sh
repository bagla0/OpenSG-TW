#!/usr/bin/env bash
set -e
SRC=~/claude_tmp/papercompile
DST=~/claude_tmp/xsec_paper
# base = RM taper paper main.tex -> becomes the cross-section paper
cp $SRC/main.tex $DST/main_rmbase.tex
# theory frame figure (coord2.png) + any author assets
mkdir -p $DST/author/figures
[ -f $SRC/author/coord2.png ] && cp $SRC/author/coord2.png $DST/author/ 2>/dev/null || true
echo "=== author figures referenced by RM taper ==="
grep -oE 'author/[^} ]+\.png' $SRC/main.tex | sort -u
echo "=== author dir contents (RM taper) ==="
ls -R $SRC/author 2>/dev/null | head -40
echo "=== section headers of RM taper (to plan conversion) ==="
grep -nE '\\(section|subsection)\{' $SRC/main.tex
