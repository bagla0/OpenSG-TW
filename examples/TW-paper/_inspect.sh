#!/usr/bin/env bash
cd ~/OpenSG-TW-claude/examples/TW-paper
echo "=== lib/ ==="; ls lib/
echo "=== iea22_blade/ ==="; ls -R iea22_blade/ 2>/dev/null | head -40
echo "=== single_cell_tube/ files ==="; ls single_cell_tube/
echo "=== two_cell_tube/data C6 + wc ==="; ls two_cell_tube/data/ | grep -iE 'C6|wc|solid_tube2cell_thin|tube2cell_thin.yaml'
echo "=== any run/common scripts ==="; ls */*.py 2>/dev/null | head -30
echo "=== single_cell shell_center head ==="; head -12 single_cell_tube/data/shell_center.yaml
echo "=== C6_solid_314 ==="; cat single_cell_tube/data/C6_solid_314.txt 2>/dev/null | head -8
