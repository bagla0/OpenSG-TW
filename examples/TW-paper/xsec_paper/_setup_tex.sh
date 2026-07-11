#!/usr/bin/env bash
set -e
SRC=~/claude_tmp/papercompile
DST=~/claude_tmp/xsec_paper
mkdir -p $DST/figures
# reuse the compile-proven class + packages + bib style from the tapered paper
for f in elsarticle.cls elsarticle-num.bst elsarticle-harv.bst multirow.sty bigdelim.sty \
         bigstrut.sty algorithmicx.sty algpseudocode.sty; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/" 2>/dev/null || true
done
echo "=== bib source present? ==="
ls $SRC/*.bib 2>/dev/null && cp $SRC/*.bib $DST/ 2>/dev/null || echo "no .bib (uses .bbl)"
[ -f $SRC/main.bbl ] && cp $SRC/main.bbl $DST/refs_from_tapered.bbl && echo "copied .bbl"
# figures from the xsec figure run
cp ~/OpenSG-TW-claude/examples/TW-paper/xsec_paper/figures/*.png $DST/figures/ 2>/dev/null && echo "copied figures"
echo "=== DST contents ==="; ls $DST; echo "--- figures ---"; ls $DST/figures
