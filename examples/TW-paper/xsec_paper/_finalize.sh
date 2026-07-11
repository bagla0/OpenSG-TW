#!/usr/bin/env bash
cd ~/claude_tmp/xsec_paper
pdflatex -interaction=nonstopmode main.tex > c1.log 2>&1
pdflatex -interaction=nonstopmode main.tex > c2.log 2>&1
grep 'Output written' c2.log
# self-contained Overleaf zip (now includes tab/ table fragments)
rm -f rm_xsec_overleaf.zip
zip -qr rm_xsec_overleaf.zip main.tex elsarticle.cls elsarticle-num.bst elsarticle-harv.bst \
    multirow.sty bigdelim.sty bigstrut.sty algorithmicx.sty algpseudocode.sty figures tab
echo "zip:"; ls -la rm_xsec_overleaf.zip main.pdf
# commit the paper's compute package to OpenSG-TW
cd ~/OpenSG-TW-claude
git add examples/TW-paper/xsec_paper/ examples/TW-paper/lib examples/TW-paper/iea22_blade/data 2>/dev/null
git commit -q -m "RM cross-section paper: full-blade RM-6DOF-vs-solid, r=0.2 timing (shell/JAX/FEniCS), full 6x6 IEA tables, PreVABS no-edge figures, ellipse 4-cell, theory+refs" 2>&1 | head -2
git log --oneline -1
