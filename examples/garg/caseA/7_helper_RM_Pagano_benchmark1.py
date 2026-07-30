"""Benchmark 1 -- caseA [0/90/0] Pagano graphite/epoxy, S = a/h in {4, 10}.

MSG-RM (mid-surface 8x8, FF integrated from the exact solution, example-7 recovery)
vs the exact 3-D elasticity and Garg's FSDT (k = 5/6) baseline, at the standard
stations x = 0 (sigma13) and x = a/2 (sigma11, sigma33).  Writes pagano_S<S>.dat and
pagano_S<S>.png into this folder.

Run:  python examples/garg/caseA/7_helper_RM_Pagano_benchmark1.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pagano_bench import run_benchmark

run_benchmark("caseA", (4, 10), tag="1")
