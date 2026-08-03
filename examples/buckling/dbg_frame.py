"""dbg_frame.py -- is the wall frame actually used by solid_aniso_sq.py right-handed and SPD?

dbg_plyC proved rot_global is valid for a PROPER rotation (det=+1).  It did NOT test the frame the solid
assembly really builds: e1=x, e2=contour tangent, e3=-inward_normal.  If e1 x e2 = -e3 the frame is
LEFT-handed (a reflection, det=-1), and the Bond transform is not a rotation -- the off-diagonal shear
couplings (C16,C26,C45) flip sign inconsistently, which mirrors a -45 ply into +45 and can break SPD.
Pure numpy: no FEM, runs instantly.
"""
import numpy as np

MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
NPER = 64


def sq_ring(a, n):
    nps = n // 4
    cor = [(-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2)]
    P, Nin, Tg = [], [], []
    for k in range(4):
        P0 = np.array(cor[k], float); P1 = np.array(cor[(k + 1) % 4], float)
        d = (P1 - P0) / np.linalg.norm(P1 - P0); nrm = np.array([-d[1], d[0]]); mid = 0.5 * (P0 + P1)
        if np.dot(-mid, nrm) < 0:
            nrm = -nrm
        for j in range(nps):
            P.append(P0 + (j / nps) * (P1 - P0)); Nin.append(nrm); Tg.append(d)
    return np.array(P), np.array(Nin), np.array(Tg)


def ply_C(mat, ang_deg):
    E1, E2, G12 = mat["E1"], mat["E2"], mat["G12"]; nu12 = mat["nu12"]
    E3 = E2; G13 = G12; G23 = 0.4 * E2; nu13 = nu12; nu23 = 0.4
    nu21 = nu12 * E2 / E1; nu31 = nu13 * E3 / E1; nu32 = nu23 * E3 / E2
    S = np.array([[1 / E1, -nu21 / E2, -nu31 / E3, 0, 0, 0],
                  [-nu12 / E1, 1 / E2, -nu32 / E3, 0, 0, 0],
                  [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
                  [0, 0, 0, 1 / G23, 0, 0], [0, 0, 0, 0, 1 / G13, 0], [0, 0, 0, 0, 0, 1 / G12]])
    C = np.linalg.inv(S)
    c, s = np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))
    Tm = np.array([[c*c, s*s, 0, 0, 0, 2*c*s], [s*s, c*c, 0, 0, 0, -2*c*s], [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, c, -s, 0], [0, 0, 0, s, c, 0], [-c*s, c*s, 0, 0, 0, c*c - s*s]])
    R = np.diag([1., 1., 1., 2., 2., 2.])
    return np.linalg.inv(Tm) @ C @ R @ Tm @ np.linalg.inv(R)


def rot_global(Cl, e1, e2, e3):
    A = np.array([e1, e2, e3])
    K1 = A**2
    K2 = np.array([[A[i, 1]*A[i, 2], A[i, 2]*A[i, 0], A[i, 0]*A[i, 1]] for i in range(3)])
    K3 = np.array([[A[1, j]*A[2, j], A[2, j]*A[0, j], A[0, j]*A[1, j]] for j in range(3)]).T
    K4 = np.array([[A[1, (j+1) % 3]*A[2, (j+2) % 3] + A[1, (j+2) % 3]*A[2, (j+1) % 3],
                    A[2, (j+1) % 3]*A[0, (j+2) % 3] + A[2, (j+2) % 3]*A[0, (j+1) % 3],
                    A[0, (j+1) % 3]*A[1, (j+2) % 3] + A[0, (j+2) % 3]*A[1, (j+1) % 3]] for j in range(3)]).T
    Tt = np.block([[K1, 2 * K2], [K3, K4]])
    Rm = np.diag([1., 1., 1., 2., 2., 2.])
    return np.linalg.inv(Tt) @ Cl @ Rm @ Tt @ np.linalg.inv(Rm)


_, nin0, tg0 = sq_ring(1.0, NPER)
Cloc = ply_C(MAT, -45.0)
print("wall frames actually used by solid_aniso_sq.py  (e1=x, e2=tangent, e3=-nin)")
print("   j   face      det(A)   e1xe2.e3    min_eig(C_global)   status")
bad_h = bad_s = 0
seen = set()
for j in range(NPER):
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, tg0[j][0], tg0[j][1]])
    e3 = np.array([0.0, -nin0[j][0], -nin0[j][1]])
    A = np.array([e1, e2, e3]); det = np.linalg.det(A); dot = np.dot(np.cross(e1, e2), e3)
    Cg = rot_global(Cloc, e1, e2, e3)
    ev = np.linalg.eigvalsh(0.5 * (Cg + Cg.T)).min()
    if det < 0:
        bad_h += 1
    if ev <= 0:
        bad_s += 1
    face = j // (NPER // 4)
    if face not in seen:                      # one representative row per face
        seen.add(face)
        print("   %2d   %d      %+.3f     %+.3f      %+.4e        %s"
              % (j, face, det, dot, ev, "OK" if (det > 0 and ev > 0) else "*** BAD ***"))
print("\n   left-handed frames : %d / %d" % (bad_h, NPER))
print("   non-SPD  C_global  : %d / %d" % (bad_s, NPER))

# What a reflection does to the ply: compare the LEFT-handed result against the properly right-handed one.
print("\nif left-handed, the fix is e3 = +e1 x e2.  Effect on the -45 ply:")
j = 0
e1 = np.array([1.0, 0.0, 0.0]); e2 = np.array([0.0, tg0[j][0], tg0[j][1]])
e3_used = np.array([0.0, -nin0[j][0], -nin0[j][1]])
e3_rh = np.cross(e1, e2)
Cu = rot_global(Cloc, e1, e2, e3_used); Cr = rot_global(Cloc, e1, e2, e3_rh)
print("   e3 used = %s   e3 right-handed = %s" % (e3_used, e3_rh))
print("   max rel diff C(used) vs C(right-handed) = %.3e" % (np.max(np.abs(Cu - Cr)) / np.max(np.abs(Cr))))
for nm, Cx in (("used", Cu), ("right-handed", Cr)):
    print("   %-13s C16=%+.4e C26=%+.4e C45=%+.4e  min_eig=%+.4e"
          % (nm, Cx[0, 5], Cx[1, 5], Cx[3, 4], np.linalg.eigvalsh(0.5 * (Cx + Cx.T)).min()))
