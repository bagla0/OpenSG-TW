"""analytic_cylinder.py -- CLOSED-FORM buckling of an anisotropic circular cylinder, and an analytical
prediction of the FSM's anisotropic error.  No FEM anywhere in this file.

=============================== DERIVATION, STEP BY STEP ===============================

Circular cylindrical shell: radius R, thickness h, laminate [[A,B],[B,D]], axial compression.
A SINGLE ply is homogeneous through the thickness, so B = 0 and the Airy-function reduction is exact.

STEP 1  Donnell kinematics.  x axial, y = R*theta circumferential, w radial.
        eps_x = u,x ;  eps_y = v,y + w/R ;  gam_xy = u,y + v,x
        kap_x = -w,xx ; kap_y = -w,yy ; kap_xy = -2 w,xy

STEP 2  Airy stress function F:  N_x = F,yy ,  N_y = F,xx ,  N_xy = -F,xy  (equilibrium satisfied
        identically).  With a = inv(A), the compatibility eps_x,yy + eps_y,xx - gam_xy,xy = -w,xx / R gives

           a22 F,xxxx - 2 a26 F,xxxy + (2 a12 + a66) F,xxyy - 2 a16 F,xyyy + a11 F,yyyy = -w,xx / R

STEP 3  Moment equilibrium with the pre-buckling axial resultant:

           D11 w,xxxx + 4 D16 w,xxxy + 2(D12+2D66) w,xxyy + 4 D26 w,xyyy + D22 w,yyyy
               + (1/R) F,xx + N_x w,xx = 0

STEP 4  THE MODE.  For a closed cylinder the general harmonic mode is the HELIX

           w = W exp( i (k x + q y) ),   k = m*pi/L,   q = n/R,   n integer, EITHER SIGN

        Every fourth-order derivative is real for this form:
           w,xxxx -> k^4 ,  w,xxxy -> k^3 q ,  w,xxyy -> k^2 q^2 ,  w,xyyy -> k q^3 ,  w,yyyy -> q^4
        so define
           Dhat(k,q) = D11 k^4 + 4 D16 k^3 q + 2(D12+2D66) k^2 q^2 + 4 D26 k q^3 + D22 q^4
           Ahat(k,q) = a22 k^4 - 2 a26 k^3 q + (2 a12 + a66) k^2 q^2 - 2 a16 k q^3 + a11 q^4

STEP 5  Eliminating F between steps 2 and 3 gives the closed form for the axial resultant at buckling

           N(k,q) = Dhat(k,q) / k^2  +  k^2 / ( R^2 * Ahat(k,q) )                          (*)

        SANITY: isotropic gives Dhat = D (k^2+q^2)^2 and Ahat = (k^2+q^2)^2/(E h), so with
        s = (k^2+q^2)^2/k^2,  N = D s + E h /(R^2 s), minimised at N = 2 sqrt(D E h)/R
        = E h^2 / (R sqrt(3(1-nu^2))) -- the classical cylinder formula.  (Checked numerically below.)

STEP 6  *** THE ANISOTROPIC POINT ***
        Dhat and Ahat contain terms ODD in q: 4*D16 k^3 q, 4*D26 k q^3, -2*a26 k^3 q, -2*a16 k q^3.
        Therefore

           Dhat(k,-q) != Dhat(k,q)   and   Ahat(k,-q) != Ahat(k,q)   whenever D16,D26,a16,a26 != 0

        i.e. THE TWO HELICAL SENSES (right- and left-handed) BUCKLE AT DIFFERENT LOADS.  The true
        critical load takes the WEAKER helix:

           N_true = min over (m, n of EITHER sign) of N(k,q)

STEP 7  *** WHAT THE FSM IS FORCED TO DO ***
        The FSM assumes w = W(y) sin(k x): one axial phase, real amplitude.  A standing wave is an
        EQUAL mixture of the two helices,
           sin(kx) cos(qy) = 1/2 [ sin(kx + qy) + sin(kx - qy) ],
        and the two helices are orthogonal, so the Rayleigh quotient of the mixture is the ratio of summed
        energies to summed geometric terms.  Both helices share |k|, hence the same geometric term, giving

           N_FSM(k,q) = 1/2 [ N(k,+q) + N(k,-q) ]          -- the ARITHMETIC MEAN of the two branches

        while the truth is the MINIMUM.  Since mean >= min always, with equality iff the two branches
        coincide:

           *** N_FSM >= N_true, with equality if and only if D16 = D26 = a16 = a26 = 0 ***

        This PROVES analytically that (a) the FSM is exact for orthotropic laminates, and (b) it must
        OVER-predict for any laminate with 16/26 coupling -- the sign we measured.  And it is quantitative:
        the predicted error is mean/min of the two helical branches, computed below with no free parameters.
========================================================================================
"""
import numpy as np

E1, E2, G12, NU12 = 140e9, 10e9, 5e9, 0.3
T, R, L = 0.02, 1.0, 2.0
E_ISO, NU_ISO = 200e9, 0.3
MMAX, NMAX = 40, 60


def qbar(ang):
    """rotated reduced stiffness of one ply"""
    nu21 = NU12 * E2 / E1; den = 1 - NU12 * nu21
    Q = np.array([[E1 / den, NU12 * E2 / den, 0.0], [NU12 * E2 / den, E2 / den, 0.0], [0.0, 0.0, G12]])
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    T1 = np.array([[c*c, s*s, 2*c*s], [s*s, c*c, -2*c*s], [-c*s, c*s, c*c - s*s]])
    Rm = np.diag([1.0, 1.0, 2.0])
    return np.linalg.inv(T1) @ Q @ Rm @ T1 @ np.linalg.inv(Rm)


def AD_single(ang):
    Qb = qbar(ang)
    return Qb * T, Qb * T ** 3 / 12.0            # B = 0 for a homogeneous single ply


def AD_iso():
    C = E_ISO / (1 - NU_ISO ** 2)
    Qm = C * np.array([[1, NU_ISO, 0], [NU_ISO, 1, 0], [0, 0, (1 - NU_ISO) / 2]])
    return Qm * T, Qm * T ** 3 / 12.0


def branches(A, D):
    """N(k,q) on the (m,n) grid for q>0 and q<0 separately.  Returns (Npos, Nneg) arrays."""
    a = np.linalg.inv(A)
    a11, a12, a16, a22, a26, a66 = a[0, 0], a[0, 1], a[0, 2], a[1, 1], a[1, 2], a[2, 2]
    D11, D12, D16, D22, D26, D66 = D[0, 0], D[0, 1], D[0, 2], D[1, 1], D[1, 2], D[2, 2]
    m = np.arange(1, MMAX + 1)[:, None]
    n = np.arange(0, NMAX + 1)[None, :]
    k = m * np.pi / L
    out = []
    for sgn in (+1.0, -1.0):
        q = sgn * n / R
        Dh = (D11 * k**4 + 4 * D16 * k**3 * q + 2 * (D12 + 2 * D66) * k**2 * q**2
              + 4 * D26 * k * q**3 + D22 * q**4)
        Ah = (a22 * k**4 - 2 * a26 * k**3 * q + (2 * a12 + a66) * k**2 * q**2
              - 2 * a16 * k * q**3 + a11 * q**4)
        with np.errstate(divide="ignore", invalid="ignore"):
            N = Dh / k**2 + k**2 / (R**2 * Ah)
        N = np.where((Ah > 0) & (Dh > 0) & np.isfinite(N), N, np.inf)
        out.append(N)
    return out


def analyse(label, A, D):
    Np, Nn = branches(A, D)
    true = min(Np.min(), Nn.min())                        # weaker helix wins
    mean = 0.5 * (Np + Nn)                                # FSM: forced equal mixture
    fsm = mean.min()
    P = lambda N: 2 * np.pi * R * N                       # resultant -> total axial force
    i = np.unravel_index(np.argmin(np.minimum(Np, Nn)), Np.shape)
    a = np.linalg.inv(A)
    print("   %-11s  D16/D11=%+7.4f  a16/a11=%+7.4f  |  N_true=%.5e  N_FSM=%.5e  ratio=%7.4f"
          % (label, D[0, 2] / D[0, 0], a[0, 2] / a[0, 0], true, fsm, fsm / true))
    print("                 critical (m,n)=(%d,%d)   P_true=%.5e N   P_FSM=%.5e N"
          % (i[0] + 1, i[1], P(true), P(fsm)))
    return fsm / true, P(true), P(fsm)


print(__doc__.split("=" * 39)[0].strip()[:0])            # keep stdout clean; derivation lives in the file
print("ANALYTICAL anisotropic cylinder buckling -- no FEM\n")
print("   R=%.1f m  t=%.3f m  L=%.1f m   E1=%.0f E2=%.0f G12=%.0f GPa  nu12=%.2f\n"
      % (R, T, L, E1 / 1e9, E2 / 1e9, G12 / 1e9, NU12))

Ai, Di = AD_iso()
den = np.sqrt(3 * (1 - NU_ISO ** 2))
classical = E_ISO * T ** 2 / (R * den)
Np, Nn = branches(Ai, Di)
print("   STEP 5 SANITY (isotropic, E=%.0f GPa):" % (E_ISO / 1e9))
print("      closed form  E t^2 /(R sqrt(3(1-nu^2))) = %.5e N/m" % classical)
print("      grid minimum of N(k,q)                  = %.5e N/m   ratio %.5f\n"
      % (min(Np.min(), Nn.min()), min(Np.min(), Nn.min()) / classical))

print("   STEP 6-7  two helical branches, and the mean-vs-min penalty the FSM must pay:")
MEAS = {0.0: 1.0050, 90.0: 1.0865, -30.0: 1.5766, -45.0: 1.3715, 45.0: 1.3700, -60.0: 1.1572}
rows = []
for ang in (0.0, 90.0, -30.0, -45.0, 45.0, -60.0):
    A, D = AD_single(ang)
    r, pt, pf = analyse("ply %+.0f" % ang, A, D)
    rows.append((ang, r, MEAS[ang]))

print("\n   PREDICTION vs MEASUREMENT (measured = FSM/3-D solid, from test_aniso_prismatic.py):")
print("      ply     analytic mean/min    measured FSM/solid")
for ang, r, meas in rows:
    print("      %+5.0f        %8.4f             %8.4f" % (ang, r, meas))
print("""
   If the analytic mean/min column reproduces the measured column, the mechanism is PROVEN: the FSM error
   is exactly the penalty for being forced onto an equal mixture of two helices that do not buckle at the
   same load.  If it predicts ~1.0 where we measure 1.37, the helix argument is NOT the explanation and
   something else is wrong.  Either way this is decided analytically, with no free parameters.
""")
