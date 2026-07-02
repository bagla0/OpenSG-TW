"""Verify omega_3 (drilling rotation fluctuation) is computed correctly:

    omega_3 = (1/C33) ( phi3 - e3^T C^ab (theta + e_b omega_b) )

with phi3 the DRILLING ANGLE (thesis eq 4.11, from the in-plane symmetry
eps12=eps21), and check it reduces to the appendix eq:omega3 (thesis 4.19):

    omega_3 = 1/(2 C33) [ k11 (X11 Rn2 - X12 Rn1)
                          + w_i' (X11 Xi2 - X12 Xi1)
                          + (w_{i|1} Xi2 - w_{i|2} Xi1) ]  -  (C_{3b}/C33) omega_b

Transcribed from 'Omega3 derive.pdf' (2 pp). k11=torsion kappa_1, k12/k13=bending
kappa_2/3, th1=twist theta_1.
"""
import sympy as sp

X11, X12, X21, X22, X31, X32 = sp.symbols('X11 X12 X21 X22 X31 X32')
x2, x3 = sp.symbols('x2 x3')
u1p, u2p, u3p = sp.symbols('u1p u2p u3p')          # u_i'   (beam-disp gradients)
w1p, w2p, w3p = sp.symbols('w1p w2p w3p')          # w_i'   (warp axial deriv)
w1_1, w2_1, w3_1 = sp.symbols('w1_1 w2_1 w3_1')    # w_{i|1}
w1_2, w2_2, w3_2 = sp.symbols('w1_2 w2_2 w3_2')    # w_{i|2}
th1 = sp.symbols('th1')                            # theta_1  (twist)
k11, k12, k13 = sp.symbols('k11 k12 k13')          # kappa_1i (torsion, bend2, bend3)
om1, om2 = sp.symbols('om1 om2')                   # omega_1, omega_2

Xi = {1: {1: X11, 2: X12}, 2: {1: X21, 2: X22}, 3: {1: X31, 2: X32}}
up = {1: u1p, 2: u2p, 3: u3p}; wp = {1: w1p, 2: w2p, 3: w3p}
wm = {(1, 1): w1_1, (2, 1): w2_1, (3, 1): w3_1, (1, 2): w1_2, (2, 2): w2_2, (3, 2): w3_2}

C31 = X21 * X32 - X31 * X22
C32 = X12 * X31 - X11 * X32
C33 = X11 * X22 - X21 * X12
Rn1 = x2 * X31 - x3 * X21
Rn2 = x2 * X32 - x3 * X22


def Grow(i, a):
    """row-i integrand of phi3 (from Omega3 derive.pdf p1)."""
    xa1 = Xi[1][a]
    base = up[i] * xa1 + wp[i] * xa1 + wm[(i, a)]
    if i == 1:
        base += -Xi[3][a] * u3p - Xi[2][a] * u2p + xa1 * x3 * k12 - xa1 * x2 * k13
    elif i == 2:
        base += -Xi[3][a] * th1 - xa1 * x3 * k11
    else:
        base += Xi[2][a] * th1 + xa1 * x2 * k11
    return base


phi3 = (sp.Rational(1, 2) * sum(Xi[i][2] * Grow(i, 1) for i in (1, 2, 3))
        - sp.Rational(1, 2) * sum(Xi[i][1] * Grow(i, 2) for i in (1, 2, 3)))

e3term = C31 * (th1 + om1) + C32 * (-u3p + om2) + C33 * u2p     # e3^T C^ab (theta + e_b omega_b)
omega3 = sp.expand((phi3 - e3term) / C33)

target = (k11 * (X11 * Rn2 - X12 * Rn1)
          + sum(wp[i] * (X11 * Xi[i][2] - X12 * Xi[i][1]) for i in (1, 2, 3))
          + sum(wm[(i, 1)] * Xi[i][2] - wm[(i, 2)] * Xi[i][1] for i in (1, 2, 3))) / (2 * C33) \
    - (C31 * om1 + C32 * om2) / C33

diff = sp.simplify(omega3 - target)
print("omega3(from phi3) - target(eq 4.19) simplify =", diff)
assert diff == 0, "MISMATCH"
print("[PASS] omega_3 from the drilling angle phi3 == appendix eq:omega3 (thesis 4.19)")

# show the cancellations explicitly
print("\ncoefficient checks (should confirm the drilling structure):")
print("  theta_1 coeff in phi3 :", sp.factor(sp.expand(phi3).coeff(th1, 1)), " (== C31)")
print("  theta_1 in e3term     :", sp.expand(e3term).coeff(th1, 1), " (cancels)")
print("  u_i' survive in omega3?", any(sp.expand(omega3).coeff(s, 1) != 0 for s in (u1p, u2p, u3p)))
print("  k12,k13 survive?       ", any(sp.expand(omega3).coeff(s, 1) != 0 for s in (k12, k13)))
print("  k11 coeff in omega3    :", sp.factor(sp.expand(omega3).coeff(k11, 1) * (2 * C33)),
      "\n                          target:", sp.factor((X11 * Rn2 - X12 * Rn1)))
