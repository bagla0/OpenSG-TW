#!/usr/bin/env bash
set -e
cd ~/claude_tmp/xsec_paper
# self-contained Overleaf project: main.tex + class/styles/bst + figures (+ PDF for reference)
rm -f rm_xsec_overleaf.zip
zip -q rm_xsec_overleaf.zip main.tex elsarticle.cls elsarticle-num.bst elsarticle-harv.bst \
    multirow.sty bigdelim.sty bigstrut.sty algorithmicx.sty algpseudocode.sty figures/*.png
echo "=== zip contents ==="; unzip -l rm_xsec_overleaf.zip | tail -20
ls -la rm_xsec_overleaf.zip main.pdf
