"""dbg_plyC.py -- is the rotated ply stiffness a VALID stiffness?

The anisotropic solid returned lam1=4.97e4 vs 3.03e7 isotropic (600x low) -- the signature of a spurious
near-zero-energy mode, which is what a NON-positive-definite C produces.  Checks, in order of severity:
  1. symmetry of C
  2. positive definiteness (all eigenvalues > 0) -- a stiffness MUST be SPD
  3. rotation by 0 deg must return the unrotated C
  4. an ISOTROPIC input must be invariant under ANY rotation (the sharpest test of the Bond transform)
  5. frame rotation must preserve eigenvalues (orthogonal similarity of the energy)
"""
import numpy as np
import sys

MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)


def ply_C(mat, ang_deg):                       # copied verbatim from solid_aniso_sq.py (import would re-run it)
    E1, E2, G12 = mat["E1"], mat["E2"], mat["G12"]; nu12 = mat["nu12"]
    E3 = E2; G13 = G12; G23 = 0.4 * E2; nu13 = nu12; nu23 = 0.4
    nu21 = nu12 * E2 / E1; nu31 = nu13 * E3 / E1; nu32 = nu23 * E3 / E2
    S = np.array([[1 / E1, -nu21 / E2, -nu31 / E3, 0, 0, 0],
                  [-nu12 / E1, 1 / E2, -nu32 / E3, 0, 0, 0],
                  [-nu13 / E1, -nu23 / E2, 1 / E3, 0, 0, 0],
                  [0, 0, 0, 1 / G23, 0, 0],
                  [0, 0, 0, 0, 1 / G13, 0],
                  [0, 0, 0, 0, 0, 1 / G12]])
    C = np.linalg.inv(S)
    c, s = np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))
    Tm = np.array([[c*c, s*s, 0, 0, 0, 2*c*s],
                   [s*s, c*c, 0, 0, 0, -2*c*s],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, c, -s, 0],
                   [0, 0, 0, s, c, 0],
                   [-c*s, c*s, 0, 0, 0, c*c - s*s]])
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

np.set_printoptions(precision=3, suppress=True, linewidth=140)
ok = True


def chk(name, cond):
    global ok
    ok &= bool(cond); print("   %-52s %s" % (name, "OK" if cond else "*** FAIL ***"))


print("1/2) ply_C symmetry + positive definiteness")
for ang in (0.0, -45.0, 45.0, 30.0):
    C = ply_C(MAT, ang)
    sym = np.max(np.abs(C - C.T)) / np.max(np.abs(C))
    ev = np.linalg.eigvalsh(0.5 * (C + C.T))
    print("   ang=%+6.1f  sym_err=%.2e  min_eig=%.4e  max_eig=%.4e" % (ang, sym, ev.min(), ev.max()))
    chk("  ang=%+.0f symmetric" % ang, sym < 1e-10)
    chk("  ang=%+.0f positive definite" % ang, ev.min() > 0)

print("\n3) rotation by 0 deg returns the unrotated matrix")
C0 = ply_C(MAT, 0.0)
e1, e2, e3 = np.eye(3)
chk("rot_global(identity frame) == C", np.max(np.abs(rot_global(C0, e1, e2, e3) - C0)) < 1e-6 * np.max(np.abs(C0)))

print("\n4) ISOTROPIC material must be invariant under any rotation (sharpest Bond-transform test)")
E, nu = 200e9, 0.3
lm = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
Ciso = np.zeros((6, 6))
Ciso[:3, :3] = lm; Ciso[0, 0] = Ciso[1, 1] = Ciso[2, 2] = lm + 2 * mu
Ciso[3, 3] = Ciso[4, 4] = Ciso[5, 5] = mu
th = np.radians(37.0)
a1 = np.array([np.cos(th), np.sin(th), 0.0]); a2 = np.array([-np.sin(th), np.cos(th), 0.0]); a3 = np.cross(a1, a2)
Cr = rot_global(Ciso, a1, a2, a3)
err = np.max(np.abs(Cr - Ciso)) / np.max(np.abs(Ciso))
print("   max rel change under 37 deg rotation = %.3e" % err)
chk("isotropic C invariant under rotation", err < 1e-8)

print("\n5) rotated anisotropic C must stay SPD and keep its eigenvalues")
Cm = ply_C(MAT, -45.0)
Cg = rot_global(Cm, a1, a2, a3)
ev0 = np.sort(np.linalg.eigvalsh(0.5 * (Cm + Cm.T)))
ev1 = np.sort(np.linalg.eigvalsh(0.5 * (Cg + Cg.T)))
print("   eig(before) = %s" % np.array2string(ev0, precision=3))
print("   eig(after ) = %s" % np.array2string(ev1, precision=3))
chk("rotated C still positive definite", ev1.min() > 0)
chk("rotated C symmetric", np.max(np.abs(Cg - Cg.T)) / np.max(np.abs(Cg)) < 1e-10)

print("\nRESULT: %s" % ("all checks passed" if ok else "*** the rotated stiffness is INVALID ***"))
sys.exit(0 if ok else 1)
