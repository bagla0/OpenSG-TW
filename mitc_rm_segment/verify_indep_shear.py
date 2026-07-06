"""verify_indep_shear.py -- verify the PAPER's independent-drilling transverse-shear
Gamma rows against the FULL eliminated analytical expression (opensg-rm-timo Eq. 2eps13/23).

Identity under test (pointwise algebra, generic tapered point):
    2gamma_a3(independent rows, omega_3 = omega_3^elim)  ==  2gamma_a3(eliminated, exact 1/C33)
with omega_3^elim from eq:om3.  Uses the exact cone geometry + analytic random fields of
verify_strains_paper (no FE, no interpolation): must agree to ~1e-12.

Also checks each Gamma_eps / Gamma_h / Gamma_l shear coefficient individually.
"""
import math
import numpy as np

# ---- reuse the exact cone rig of verify_strains_paper (same constants) ----
R0, m = 1.0, -0.15
z0, th0 = 1.0, 0.7
AX, CROSS = 2, (0, 1)


def Rz(z):
    return R0 + m * z


def point(th, z):
    return np.array([Rz(z) * math.cos(th), Rz(z) * math.sin(th), z])


PHI = math.radians(25.0)   # in-surface frame rotation: makes x_{1;2}=a2.b1 NONZERO,
                           # exercising every x12-carrying term (on a straight-axis tube
                           # the natural hoop has x12=0 identically, hiding those terms)


def frame(th):
    a2c = np.array([-math.sin(th), math.cos(th), 0.0])
    g = np.array([m * math.cos(th), m * math.sin(th), 1.0])
    a1c = g / np.linalg.norm(g)
    a1 = math.cos(PHI) * a1c - math.sin(PHI) * a2c
    a2 = math.sin(PHI) * a1c + math.cos(PHI) * a2c
    n = np.cross(a1, a2)
    return a1, a2, n


def beam_comp(v):
    return np.array([v[AX], v[CROSS[0]], v[CROSS[1]]])


S1F = 1.0 / math.sqrt(1.0 + m * m)
H = 1e-6


def dhoop(f, th=th0, z=z0):
    return (f(th + H, z) - f(th - H, z)) / (2 * H) / Rz(z)


def dgen(f, th=th0, z=z0):
    return (f(th, z + H) - f(th, z - H)) / (2 * H) * S1F


def dz1(f, th=th0, z=z0):   # derivative along the ROTATED a1
    return math.cos(PHI) * dgen(f, th, z) - math.sin(PHI) * dhoop(f, th, z)


def dz2(f, th=th0, z=z0):   # derivative along the ROTATED a2
    return math.sin(PHI) * dgen(f, th, z) + math.cos(PHI) * dhoop(f, th, z)


a1, a2, nn = frame(th0)
xi1 = beam_comp(a1); xi2 = beam_comp(a2); yv = beam_comp(nn)
P = beam_comp(point(th0, z0))
x2, x3 = P[1], P[2]
x11, x21, x31 = xi1
x12, x22, x32 = xi2
y1, y2, y3 = yv
C33 = y3
Rn1 = x2 * x31 - x3 * x21
Rn2 = x2 * x32 - x3 * x22

# analytic random fields (same family as verify_strains_paper)
RNG = np.random.default_rng(7)
CW = RNG.normal(size=(3, 5)); CO = RNG.normal(size=(2, 5))
CWp = RNG.normal(size=(3, 5))
EB = np.array([0.11, -0.23, 0.17, 0.09])          # [g11, k1, k2, k3]


def sfield(C, th, z):
    b = np.array([math.sin(2 * th), math.cos(th), (z - z0), (z - z0) ** 2, (th - th0) * (z - z0)])
    return C @ b


def w_f(th, z):   return sfield(CW, th, z)
def om_f(th, z):  return sfield(CO, th, z)     # omega_1, omega_2 (independent shear rotations)
def wp_f(th, z):  return sfield(CWp, th, z)    # w'_i


w0 = w_f(th0, z0); om0 = om_f(th0, z0); wp0 = wp_f(th0, z0)
w_1, w_2 = dz1(w_f), dz2(w_f)                   # w_{i|1}, w_{i|2}
g11, k1, k2, k3 = EB

# ---- eliminated omega_3 (eq:om3, exact 1/(2 C33)) ----
S_num = (k1 * (x11 * Rn2 - x12 * Rn1)
         + float(wp0 @ (x11 * xi2 - x12 * xi1))
         + float(w_1 @ xi2) - float(w_2 @ xi1))
om3_elim = S_num / (2.0 * C33) - (y1 * om0[0] + y2 * om0[1]) / C33

# ---- FULL ELIMINATED analytical shear (verbatim opensg-rm-timo Eq. 2eps13/2eps23) ----
inv33 = 1.0 / C33
h33 = 0.5 * inv33
k113 = (x2 * ((x32 * x32 * x11 - x32 * x31 * x12) * h33 + C33 * x11)
        - x3 * ((x32 * x22 * x11 - x32 * x21 * x12) * h33 + y2 * x11))
k123 = (x2 * ((x31 * x31 * x12 - x31 * x32 * x11) * h33 + C33 * x12)
        - x3 * ((x31 * x21 * x12 - x31 * x22 * x11) * h33 + y2 * x12))
a13 = x32 * xi2 * h33 + yv
b13 = -x32 * xi1 * h33
a23 = x31 * xi1 * h33 + yv
b23 = -x31 * xi2 * h33
om13 = np.array([x12 - x32 * y1 * inv33, x22 - x32 * y2 * inv33])
om23 = np.array([-x11 + x31 * y1 * inv33, -x21 + x31 * y2 * inv33])
g13_elim = (x11 * y1 * g11 + k113 * k1 + x11 * y1 * (x3 * k2 - x2 * k3)
            + float(a13 @ w_1) + float(b13 @ w_2) + float(om13 @ om0)
            + float((x11 * a13 + x12 * b13) @ wp0))
g23_elim = (x12 * y1 * g11 + k123 * k1 + x12 * y1 * (x3 * k2 - x2 * k3)
            + float(a23 @ w_2) + float(b23 @ w_1) + float(om23 @ om0)
            + float((x12 * a23 + x11 * b23) @ wp0))

# ---- PAPER'S INDEPENDENT rows (eq:shear / Gamma matrices), omega_3 = om3_elim ----
swept = x2 * y3 - x3 * y2
omv = np.array([om0[0], om0[1], om3_elim])
xb2 = np.array([x12, x22, x32])                  # x_{b;2}
xb1 = np.array([x11, x21, x31])                  # x_{b;1}
g13_indep = (x11 * y1 * g11 + x11 * swept * k1 + x11 * y1 * (x3 * k2 - x2 * k3)
             + float(yv @ w_1) + x11 * float(yv @ wp0) + float(xb2 @ omv))
g23_indep = (x12 * y1 * g11 + x12 * swept * k1 + x12 * y1 * (x3 * k2 - x2 * k3)
             + float(yv @ w_2) + x12 * float(yv @ wp0) - float(xb1 @ omv))

print("tapered cone point: m=%.3f th0=%.2f z0=%.2f | C33=y3=%.6f  y1=%.6f" % (m, th0, z0, C33, y1))
print("omega_3 (eliminated value) = %.10f" % om3_elim)
print()
print("2gamma_13:  eliminated(full analytic) = %.14f" % g13_elim)
print("            independent(paper rows)   = %.14f" % g13_indep)
print("            |diff| = %.3e" % abs(g13_elim - g13_indep))
print("2gamma_23:  eliminated(full analytic) = %.14f" % g23_elim)
print("            independent(paper rows)   = %.14f" % g23_indep)
print("            |diff| = %.3e" % abs(g23_elim - g23_indep))

# ---- coefficient-level identities (the algebra behind the un-substitution) ----
print("\ncoefficient identities (must be ~0):")
# swept-area collapse (Omega3_new): x11 Rn2 - x12 Rn1 == -(x2 y2 + x3 y3)
print("  swept-id : %.3e" % abs((x11 * Rn2 - x12 * Rn1) + (x2 * y2 + x3 * y3)))
# k1 coefficient: eliminated k113 == x11*swept + x32 * (x11 Rn2 - x12 Rn1)/(2 C33)
print("  k1  (g13): %.3e" % abs(k113 - (x11 * swept + x32 * (x11 * Rn2 - x12 * Rn1) * h33)))
print("  k1  (g23): %.3e" % abs(k123 - (x12 * swept - x31 * (x11 * Rn2 - x12 * Rn1) * h33)))
# w_{i|1} coefficient of g13: a13 == y_i + x32 * x_{i;2}/(2C33)  (x32*om3 chain)
print("  w|1 (g13): %.3e" % np.abs(a13 - (yv + x32 * xi2 * h33)).max())
print("  w|2 (g13): %.3e" % np.abs(b13 - (-x32 * xi1 * h33)).max())
# omega coefficient: om13 == x_{b;2} restricted to b=1,2 with -x32*(C3b/C33)
print("  om  (g13): %.3e" % np.abs(om13 - (np.array([x12, x22]) - x32 * np.array([y1, y2]) * inv33)).max())
print("  om  (g23): %.3e" % np.abs(om23 - (-np.array([x11, x21]) + x31 * np.array([y1, y2]) * inv33)).max())
