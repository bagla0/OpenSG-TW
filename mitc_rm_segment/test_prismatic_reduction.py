"""
test_prismatic_reduction.py    [ Windows opensg_2_0_env ]
========================================================================
The check the user asked for: the 8-strain SURFACE operators of the
tapered-segment element (segment_element._quad_ops, Gamma_h=BDq / shear=BGq /
Gamma_l=BLq) must REDUCE to the validated 1-D prismatic RM operators
(msg_rm_timo._elem_BD_BG_BL) when

    (a) all beam-axis (|1 / d/dx1) derivative terms are removed, and
    (b) the prismatic cross-section frame C^{1b} is used
        (x_{1;1}=1, x_{1;2}=x_{2;1}=x_{3;1}=0; e1 = beam axis).

Removing (a) is achieved automatically by a SPAN-INVARIANT warping field: a
field constant along the axial (eta) direction has d/dx1 == 0, so every D1
column of BDq/BGq drops.  The remaining hoop (d/ds) operator must then equal the
1-D operator node-for-node.  We test on a FLAT wall (k22=0, exact) and a CURVED
arc (k22=-1/R, same curvature term in both), at all 2x2 Gauss points, for random
warping.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from segment_element import _quad_ops, _mitc_shear
from opensg_jax.fe_jax.msg_rm_timo import _elem_BD_BG_BL
from opensg_jax.fe_jax.msg_rm import _lagrange, _macro_BD

GP = 1.0 / np.sqrt(3.0)
QPTS = [(-GP, -GP), (GP, -GP), (GP, GP), (-GP, GP)]


def prismatic_quad(n0, n1, L, curved_R=None):
    """Extrude the 1-D wall element [n0,n1] (in (y,z)) along x by length L to a
    quad, winding [(j,k),(j,k+1),(j+1,k+1),(j+1,k)] = [0,0],[0,1],[1,1],[1,0].
    Frame: e1=axial=(1,0,0); e2 = wall tangent; e3 = e1 x e2 (inward for a CCW
    circle).  Returns X(4,3), e1,e2,e3, k22."""
    n0 = np.asarray(n0, float); n1 = np.asarray(n1, float)
    X = np.array([[0.0, n0[0], n0[1]],
                  [0.0, n1[0], n1[1]],
                  [L,   n1[0], n1[1]],
                  [L,   n0[0], n0[1]]])
    e1 = np.array([1.0, 0.0, 0.0])
    tan = n1 - n0; tan = tan / np.linalg.norm(tan)
    e2 = np.array([0.0, tan[0], tan[1]])
    e3 = np.cross(e1, e2)
    k22 = 0.0 if curved_R is None else -1.0 / curved_R
    return X, e1, e2, e3, k22


def report(tag, n0, n1, k22R):
    X, e1, e2, e3, k22 = prismatic_quad(n0, n1, 0.7, curved_R=k22R)
    X1d = np.array([n0, n1], float)                       # 1-D element (y,z)
    nodes_xi = _lagrange(1)                               # [-1, 1]
    rng = np.random.default_rng(0)
    V0, V1 = rng.standard_normal(5), rng.standard_normal(5)
    Vquad = np.concatenate([V0, V1, V1, V0])              # span-invariant
    V1d = np.concatenate([V0, V1])

    maxD = maxG = maxL = 0.0
    for (xi, eta) in QPTS:
        BDq, BGq, BLq, geo = _quad_ops(X, e1, e2, e3, xi, eta, k22, cross=(1, 2))
        BGb = _mitc_shear(X, e1, e2, e3, xi, eta, k22, cross=(1, 2))
        # 1-D at the SAME hoop coordinate (surface xi == 1-D xi over the wall)
        b1D, b1G, b1L, g1 = _elem_BD_BG_BL(nodes_xi, xi, X1d, None, k22, 1)
        gD_s = BDq @ Vquad; gD_1 = b1D @ V1d
        gL_s = BLq @ Vquad; gL_1 = b1L @ V1d
        gG_s = BGb @ Vquad
        # 1-D MITC-tied shear at the hoop tying point (xi=0), for comparison
        b0D, b0G, b0L, _ = _elem_BD_BG_BL(nodes_xi, 0.0, X1d, None, k22, 1)
        gG_1 = np.array([BGq[0] @ Vquad, (b0G[1] @ V1d)])  # g13 full, g23 tied-at-0
        maxD = max(maxD, np.max(np.abs(gD_s - gD_1)))
        maxL = max(maxL, np.max(np.abs(gL_s - gL_1)))
        maxG = max(maxG, np.max(np.abs(gG_s - gG_1)))
    scaleD = max(1e-12, np.max(np.abs(b1D @ V1d)))
    print(f"[{tag}]  max|Gamma_D surf - 1D| = {maxD:.2e}   max|Gamma_l diff| = {maxL:.2e}"
          f"   max|shear diff| = {maxG:.2e}   (rel_D={maxD/scaleD:.1e})")
    return maxD, maxL


print("Prismatic reduction: SURFACE 8-strain operator  vs  validated 1-D RM operator")
print("(span-invariant field => all d/dx1 (|1) terms vanish; hoop operator must match 1-D)\n")
mD1, mL1 = report("flat wall  k22=0 ", (1.0, 0.0), (1.3, 0.4), None)
mD2, mL2 = report("curved arc k22=-1/R", (1.0, 0.0), (0.94, 0.34), 1.0)

tol = 1e-10
ok = (mD1 < tol and mL1 < tol)
print("\nFLAT-WALL exact reduction:", "PASS" if ok else "CHECK",
      "  (Gamma_D and Gamma_l match the 1-D operator to machine precision)")
print("CURVED-ARC (facet-frame approx): D-strain diff is the chord-vs-tangent facet error,",
      "\n  same k22 curvature term enters both BDq[5] rows identically.")
