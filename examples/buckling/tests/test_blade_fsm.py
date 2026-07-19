"""test_blade_fsm.py -- REALISTIC composite blade: per-station FSM local buckling driven by NATIVE ring data
(no conformal mapping), with the statics guards deciding whether each station's answer is believable.

Expect, versus the conformal path:
  abd_ea_consistency  1.03-1.37 (conformal, growing outboard)  ->  ~1.000 (native ring)
  section_equilibrium ~1.02-1.08                               ->  ~1.00
"""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import blade_buckling as bb
import blade_fsm as bfsm
import robust_checks as rc

print("building blade (for the traction -> beam forces) ...")
bl = bb.build_blade(verbose=False)
nodes, quads = bl["nodes"], bl["quads"]
f = bb.traction_load(nodes, quads)
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
print("root FF = %s" % np.array2string(FF[0], precision=3))

B0 = bb.homogenize_station(5)
print("bundle keys: %s" % sorted([k for k in B0.keys()])[:24])
print("  has C6: %s" % ("C6" in B0))

print("\nper-station native-ring FSM (composite IEA blade):")
out, gov = bfsm.blade_scan([5, 10, 20, 30, 40], FF, verbose=True)

print("\n" + rc.report(out[2]["guards"], title="station 20 pre-flight (native ring)"))
