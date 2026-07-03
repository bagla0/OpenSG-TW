"""ROW-LEVEL verification of quad_ops_general against the paper's prismatic
strains (eq:prism).  On a PRISMATIC circular-tube quad at hoop angle theta the
operator's contour(D2)- and value(N)-coefficients of every DOF must equal the
eq:prism coefficients:

  e11    = g + x3 k2 - x2 k3                      + [w'_1]
  e22    = xd2 wd2 + xd3 wd3
  2e12   = wd1 + k1 Rn                            + [xd2 w'_2 + xd3 w'_3]
  k11    = xd2 k2 + xd3 k3   + [om'_2/xd2 - (xd3/2xd2) wd'_1 (+k'Rn: dropped)]
  k22    = -omd_1
  k12+21 = -k1 + omd_2/xd2 + (k22/2)(xd3/xd2)^2 (wd1 - k1 Rn) - k22 (xd3/xd2^2) om2
           + [-om'_1 - (xd3/xd2^2)(k22/2) w'_3 + (xd3/2xd2)(xd2 wd'_2 + xd3 wd'_3)]
  2g13   = k1 (x2(xd2 + xd3^2/2xd2) + x3 xd3/2) - wd1 xd3/(2xd2) + om2/xd2
           + [-w'_2 xd3/2 + w'_3 (xd2 + xd3^2/2xd2)]
  2g23   = (wd3 xd2 - wd2 xd3) - om1

wd = contour derivative; [..] = Gamma_l (w') parts.  Extraction: coefficient of
the CONTOUR derivative of dof q = row applied to a nodal field linear in arc-
length, centred at the Gauss point; coefficient of the VALUE = constant field.
"""
import os, sys, math
import numpy as np
sys.path.insert(0, r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code")
sys.path.insert(0, r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mitc_rm_segment")
from segment_element_general import quad_ops_general

R, k22 = 1.0, -1.0
th = 0.7                                   # test hoop angle (away from poles)
dth = 2 * math.pi / 160                    # hoop span of the quad
hz = 0.05                                  # axial span
# prismatic quad around theta, beam axis = x (cross = (1,2) -> coords (y,z))
angs = [th - dth / 2, th + dth / 2]
X = np.array([[0.0, R * math.cos(a), R * math.sin(a)] for a in angs] +
             [[hz, R * math.cos(a), R * math.sin(a)] for a in angs[::-1]])
# winding (n0,n1,n2,n3) = (th-,th+,th+,th-) axial pair -- xi ~ hoop, eta ~ axial
tm = th
e1m = np.array([1.0, 0, 0])
e2m = np.array([0, -math.sin(tm), math.cos(tm)])
e3m = np.array([0, -math.cos(tm), -math.sin(tm)])          # inward

BDe, BDh, BDl, BGe, BGh, BGl, dA = quad_ops_general(X, e1m, e2m, e3m, 0.0, 0.0, k22,
                                                    cross=(1, 2), ax=0)

# geometry at the Gauss point (element centre)
xd2, xd3 = -math.sin(th), math.cos(th)     # unit tangent (CCW)
x2, x3 = R * math.cos(th), R * math.sin(th)
Rn = x2 * xd3 - x3 * xd2

# ---- extraction helpers -------------------------------------------------------
# nodal arc-length (centred at element centre) and constants, per node
s_nod = np.array([-0.5 * R * dth, 0.5 * R * dth, 0.5 * R * dth, -0.5 * R * dth])
ONE = np.ones(4)


def coef(row, dof, kind):
    """operator coefficient of  d(dof)/ds (kind='d')  or dof value (kind='v')."""
    v = np.zeros(20)
    nod = s_nod if kind == "d" else ONE
    for a in range(4):
        v[5 * a + dof] = nod[a]
    return float(row @ v)


LBL = ["e11", "e22", "2e12", "k11", "k22s", "k12+21", "2g13", "2g23"]
rows_h = [BDh[0], BDh[1], BDh[2], BDh[3], BDh[4], BDh[5], BGh[0], BGh[1]]
rows_l = [BDl[0], BDl[1], BDl[2], BDl[3], BDl[4], BDl[5], BGl[0], BGl[1]]
rows_e = [BDe[0], BDe[1], BDe[2], BDe[3], BDe[4], BDe[5], BGe[0], BGe[1]]

# expected coefficient tables from eq:prism  (dof order w1,w2,w3,om1,om2)
Z = 0.0
exp_h_d = {  # contour-derivative coefficients
    "e11":   [Z, Z, Z, Z, Z],
    "e22":   [Z, xd2, xd3, Z, Z],
    "2e12":  [1.0, Z, Z, Z, Z],
    "k11":   [Z, Z, Z, Z, Z],
    "k22s":  [Z, Z, Z, -1.0, Z],
    "k12+21": [k22 / 2 * (xd3 / xd2) ** 2, Z, Z, Z, 1.0 / xd2],
    "2g13":  [-xd3 / (2 * xd2), Z, Z, Z, Z],
    "2g23":  [Z, -xd3, xd2, Z, Z],
}
exp_h_v = {  # value coefficients
    "e11":   [Z] * 5, "e22": [Z] * 5, "2e12": [Z] * 5, "k11": [Z] * 5, "k22s": [Z] * 5,
    "k12+21": [Z, Z, Z, Z, -k22 * xd3 / xd2 ** 2],
    "2g13":  [Z, Z, Z, Z, 1.0 / xd2],
    "2g23":  [Z, Z, Z, -1.0, Z],
}
exp_l_d = {
    "e11": [Z] * 5, "e22": [Z] * 5, "2e12": [Z] * 5,
    "k11":  [-xd3 / (2 * xd2), Z, Z, Z, Z],
    "k22s": [Z] * 5,
    "k12+21": [Z, xd3 / (2 * xd2) * xd2, xd3 / (2 * xd2) * xd3, Z, Z],
    "2g13": [Z] * 5, "2g23": [Z] * 5,
}
exp_l_v = {
    "e11":  [1.0, Z, Z, Z, Z],
    "e22":  [Z] * 5,
    "2e12": [Z, xd2, xd3, Z, Z],
    "k11":  [Z, Z, Z, Z, 1.0 / xd2],
    "k22s": [Z] * 5,
    "k12+21": [Z, Z, -(xd3 / xd2 ** 2) * (k22 / 2), -1.0, Z],
    "2g13": [Z, -xd3 / 2, xd2 + xd3 ** 2 / (2 * xd2), Z, Z],
    "2g23": [Z] * 5,
}
exp_e = {
    "e11":   [1.0, Z, x3, -x2],
    "e22":   [Z] * 4,
    "2e12":  [Z, Rn, Z, Z],
    "k11":   [Z, Z, xd2, xd3],
    "k22s":  [Z] * 4,
    "k12+21": [Z, -1.0 - (k22 / 2) * (xd3 / xd2) ** 2 * Rn, Z, Z],
    "2g13":  [Z, x2 * (xd2 + xd3 ** 2 / (2 * xd2)) + x3 * xd3 / 2, Z, Z],
    "2g23":  [Z] * 4,
}

print("row-by-row check (op vs eq:prism), theta=%.2f  xd2=%.3f xd3=%.3f" % (th, xd2, xd3))
print("%-8s %-6s %-4s %12s %12s   %s" % ("row", "block", "dof", "op", "expected", "flag"))
DOF = ["w1", "w2", "w3", "om1", "om2"]
nbad = 0
for r, name in enumerate(LBL):
    for blk, rows, table, kind in (("h.d", rows_h, exp_h_d, "d"), ("h.v", rows_h, exp_h_v, "v"),
                                   ("l.d", rows_l, exp_l_d, "d"), ("l.v", rows_l, exp_l_v, "v")):
        for q in range(5):
            got = coef(rows[r], q, kind)
            want = table[name][q]
            if abs(got - want) > 1e-6 * max(1.0, abs(want)) + 1e-8:
                nbad += 1
                print("%-8s %-6s %-4s %12.5f %12.5f   <== MISMATCH" % (name, blk, DOF[q], got, want))
    for q in range(4):
        got = float(rows_e[r][q]); want = exp_e[name][q]
        if abs(got - want) > 1e-6 * max(1.0, abs(want)) + 1e-8:
            nbad += 1
            print("%-8s %-6s eb%d %12.5f %12.5f   <== MISMATCH" % (name, "e", q, got, want))
print("\nmismatched coefficients: %d" % nbad)
