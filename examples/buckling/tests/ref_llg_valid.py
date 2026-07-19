"""ref_llg_valid.py -- do the two published validations survive the proposed _L_lg fix?"""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import shell_buckling as sb
CODE_L = sb._L_lg


def FIX_L(T):
    L = np.zeros((20, 24))
    for a in range(4):
        L[5 * a:5 * a + 3, 6 * a:6 * a + 3] = T
        L[5 * a + 3, 6 * a + 3:6 * a + 6] = T[1]
        L[5 * a + 4, 6 * a + 3:6 * a + 6] = -T[0]
    return L


for tag, Lmap in [("CODE", CODE_L), ("FIX ", FIX_L)]:
    sb._L_lg = Lmap
    fe, an = sb.validate_plate(nx=24)
    print(">>> %s plate ratio = %.4f" % (tag, fe / an))
    loads, _ = None, None
    r = sb.validate_cylinder(mesh=(120, 60), verbose=False)
    print(">>> %s cyl   ratio = %s" % (tag, r))
sb._L_lg = CODE_L
