"""reddy_hsdt_navier.py -- ANALYTICAL transient solution of the Nayak Ex.5
sandwich plate with REDDY'S THIRD-ORDER SHEAR DEFORMATION THEORY (TSDT), the
same theory Nayak-Shenoi-Moy's C0 finite elements implement.  For a simply
supported CROSS-PLY plate under the double-sine load the TSDT admits an EXACT
single-mode Navier solution (Reddy, "Mechanics of Laminated Composite Plates
and Shells", ch. 11): 5 amplitude DOFs, a 5x5 stiffness + 5x5 consistent-mass
system, integrated in CLOSED FORM per vibration mode.  A converged Nayak FE
run must reproduce this curve (his own Example 2 validates against the
analogous Khdeir-Reddy solution) -- so this script IS the "Nayak result" for
the comparison plots, with no figure digitization.

The same machinery, with the z^3 terms switched off and the OpenSG-RM 8x8
in place of the CLT integrals, also produces the ANALYTICAL OpenSG-RM (FSDT)
response -- the single-mode twin of the Abaqus S4+general-section route.

Outputs (this folder):
    reddy_wt_step.dat / reddy_wt_blast.dat   t [s], w_center [m] (TSDT), and
                                             the OpenSG-RM-Navier column
    reddy_fig13.png                          the Nayak Fig.-13 reproduction:
                                             w/0.0254 m vs t for sine, step
                                             (t1 cut-off), triangular, blast
    printed: the 5 TSDT vs RM natural frequencies + peak-w table

Run:  python examples/opensg-rm_dynamic/reddy_hsdt_navier.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE_ = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE_
while not os.path.isdir(os.path.join(ROOT, "opensg_jax")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT (the paper's Example 2/5 digits)
# ----------------------------------------------------------------------------
H = 0.1524              # plate thickness h [m] (Ex.2 value, Ex.5 inherits it)
A = 10.0 * H            # side a = b (Ex.5: a/h = 10) [m]
B = A                   # square plate
TPLY = 0.0125 * H       # one GE ply: per-face laminate 2hf/h = 0.05, 4 plies
TCORE = 0.9 * H         # HEREX C70.130 PVC foam core thickness [m]
EL, ET = 128.0e9, 11.0e9        # face: fibre / transverse moduli [Pa]
GLT, G13F, G23F = 4.48e9, 4.48e9, 1.53e9   # face shear moduli [Pa]
NULT = 0.25             # face major Poisson ratio
RHOF = 1500.0           # face density [kg/m^3]
EC, GC, NUC, RHOC = 103.63e6, 50.0e6, 0.32, 130.0   # core (isotropic)
Q0 = 68.9476e6          # load intensity q0 [Pa]
T1 = 0.006              # pulse cut-off t1 [s] (sine/step/triangular)
CBLAST = 330.0          # blast decay constant c [1/s]: F(t) = e^(-c t)
TTOT = 0.02             # response window [s]
DT_OUT = 1.0e-5         # output sampling [s]
MMODE = NMODE = 1       # Navier half-wave numbers of the load sin(pi x/a)...
PP = MMODE * np.pi / A  # p = m pi / a, the x-wavenumber
QQ = NMODE * np.pi / B  # q = n pi / b, the y-wavenumber
C1 = 4.0 / (3.0 * H * H)        # Reddy's c1: u = u0 + z*phx - c1 z^3(phx+w,x)
C3 = 3.0 * C1                   # 3 c1 = 4/h^2: gamma(z) = (1 - 3 c1 z^2)*g0
# layup, bottom -> top: (0/90/0/90/core)s, one entry per layer
THICK = [TPLY] * 4 + [TCORE] + [TPLY] * 4
ANGLES = [0.0, 90.0, 0.0, 90.0, 0.0, 90.0, 0.0, 90.0, 0.0]
ISCORE = [False] * 4 + [True] + [False] * 4
RHO = [RHOC if c else RHOF for c in ISCORE]     # per-layer density
MAT_NAMES = ["ge"] * 4 + ["herex"] + ["ge"] * 4  # for the MSG 8x8 route
MATERIAL_DB = {                                  # same digits as make_1dsg
    "ge": {"E": [EL, ET, ET], "G": [GLT, G13F, G23F],
           "nu": [NULT] * 3, "rho": RHOF},
    "herex": {"E": [EC] * 3, "G": [GC] * 3, "nu": [NUC] * 3, "rho": RHOC},
}
ZK = np.concatenate([[0.0], np.cumsum(THICK)]) - sum(THICK) / 2   # interfaces
HERE = os.path.dirname(os.path.abspath(__file__))


_EX5 = dict(H=H, A=A, B=B, THICK=list(THICK), ANGLES=list(ANGLES),
            ISCORE=list(ISCORE), RHO=list(RHO), EL=EL, ET=ET, GLT=GLT,
            G13F=G13F, G23F=G23F, NULT=NULT, RHOF=RHOF,
            MAT_NAMES=list(MAT_NAMES))     # snapshot of the Ex.5 defaults


def set_case(case):
    """Point the module configuration at one of the paper's examples.
    "ex5" = the sandwich above (the default); "ex2" = the (0/90/0) laminate
    a = b = 5h whose converged step response Nayak TABULATES (Tables 2-3)
    against the Khdeir-Reddy analytical solution -- the anchor that pins
    THIS implementation of the TSDT to the paper's own numbers."""
    global H, A, B, THICK, ANGLES, ISCORE, RHO, PP, QQ, C1, C3, ZK
    global EL, ET, GLT, G13F, G23F, NULT, RHOF, MAT_NAMES, MATERIAL_DB
    if case == "ex2":
        H = 0.1524                          # same absolute thickness
        A = B = 5.0 * H                     # Ex.2: a = b = 5h (thick!)
        THICK = [H / 3.0] * 3               # (0/90/0), equal-thickness plies
        ANGLES = [0.0, 90.0, 0.0]
        ISCORE = [False] * 3                # no core
        EL, ET = 172.369e9, 6.895e9         # the paper's Ex.2 material
        GLT, G13F, G23F = 3.448e9, 3.448e9, 1.379e9
        NULT, RHOF = 0.25, 1603.03
        RHO = [RHOF] * 3
        MAT_NAMES = ["ge"] * 3
    else:
        H, A, B = _EX5["H"], _EX5["A"], _EX5["B"]
        THICK, ANGLES = list(_EX5["THICK"]), list(_EX5["ANGLES"])
        ISCORE, RHO = list(_EX5["ISCORE"]), list(_EX5["RHO"])
        EL, ET, GLT = _EX5["EL"], _EX5["ET"], _EX5["GLT"]
        G13F, G23F = _EX5["G13F"], _EX5["G23F"]
        NULT, RHOF = _EX5["NULT"], _EX5["RHOF"]
        MAT_NAMES = list(_EX5["MAT_NAMES"])
    PP, QQ = MMODE * np.pi / A, NMODE * np.pi / B   # wavenumbers follow a,b
    C1 = 4.0 / (3.0 * H * H)                # c1 follows the thickness
    C3 = 3.0 * C1
    ZK = np.concatenate([[0.0], np.cumsum(THICK)]) - sum(THICK) / 2
    MATERIAL_DB = {
        "ge": {"E": [EL, ET, ET], "G": [GLT, G13F, G23F],
               "nu": [NULT] * 3, "rho": RHOF},
        "herex": {"E": [EC] * 3, "G": [GC] * 3, "nu": [NUC] * 3,
                  "rho": RHOC},
    }


def qbar(ang, core):
    """The plane-stress reduced stiffness Qbar (3x3, in-plane) and the
    transverse-shear Qs (2x2, rows gamma_13 / gamma_23) of one layer rotated
    to the plate axes.  Cross-ply only (0/90): the rotation is a swap."""
    if core:                                    # isotropic foam
        q11 = EC / (1.0 - NUC * NUC)            # E/(1-nu^2)
        q12 = NUC * q11                         # nu E/(1-nu^2)
        Q = np.array([[q11, q12, 0.0], [q12, q11, 0.0], [0.0, 0.0, GC]])
        Qs = np.diag([GC, GC])                  # isotropic: G13 = G23 = Gc
        return Q, Qs
    nutl = NULT * ET / EL                       # minor Poisson ratio
    den = 1.0 - NULT * nutl                     # 1 - nu12 nu21
    q11, q22 = EL / den, ET / den               # on-axis reduced stiffnesses
    q12 = NULT * ET / den                       # nu12 E2/(1 - nu12 nu21)
    Q = np.array([[q11, q12, 0.0], [q12, q22, 0.0], [0.0, 0.0, GLT]])
    Qs = np.diag([G13F, G23F])                  # on-axis (13, 23) shear
    if abs(ang - 90.0) < 1e-9:                  # 90-deg ply: axes swap
        Q = Q[[1, 0, 2]][:, [1, 0, 2]]          # 11 <-> 22 (Q66 unchanged)
        Qs = Qs[[1, 0]][:, [1, 0]]              # G13 <-> G23
    return Q, Qs


def laminate_integrals():
    """The TSDT laminate integrals: in-plane (A,B,D,E,F,H) = integral of
    Qbar*(1, z, z^2, z^3, z^4, z^6) dz and shear (As, Ds, Fs) = integral of
    Qs*(1, z^2, z^4) dz, plus the inertias I(z^0..z^6) -- all by exact
    per-layer antiderivatives of the z-powers."""
    Aab = {k: np.zeros((3, 3)) for k in (0, 1, 2, 3, 4, 6)}   # in-plane
    Ssh = {k: np.zeros((2, 2)) for k in (0, 2, 4)}            # shear
    In = np.zeros(7)                                          # inertias
    for lay in range(len(THICK)):
        Q, Qs = qbar(ANGLES[lay], ISCORE[lay])
        z0, z1 = ZK[lay], ZK[lay + 1]
        for k in Aab:                           # z^k moments, exact
            mom = (z1 ** (k + 1) - z0 ** (k + 1)) / (k + 1)
            Aab[k] += Q * mom
        for k in Ssh:
            mom = (z1 ** (k + 1) - z0 ** (k + 1)) / (k + 1)
            Ssh[k] += Qs * mom
        for k in range(7):
            In[k] += RHO[lay] * (z1 ** (k + 1) - z0 ** (k + 1)) / (k + 1)
    return Aab, Ssh, In


def navier_KM(theory):
    """The single-mode 5x5 stiffness K and consistent mass M in the amplitude
    vector d = (U, V, W, X, Y) of
        u0 = U cos(px) sin(qy),  v0 = V sin(px) cos(qy),
        w  = W sin(px) sin(qy),  phx = X cos(px) sin(qy),
        phy = Y sin(px) cos(qy)                       (SS-1, cross-ply).
    Every strain family (sinsin / coscos / cossin / sincos) integrates to
    ab/4 over the plate and cross-families vanish, so the plate energy is
    (ab/4) d^T K d -- the common ab/4 cancels against the load and is
    dropped.  theory = "tsdt" (Reddy, c1 terms on) or "fsdt" (RM: c1 = 0,
    NO shear correction factor -- the 8x8-equivalent constitutive shear)."""
    Aab, Ssh, In = laminate_integrals()
    c1 = C1 if theory == "tsdt" else 0.0        # switch the z^3 field off
    p, q = PP, QQ
    # amplitude rows of the generalized strains (signs included; each row's
    # trig family is orthogonal to the others, so only like-row products
    # survive the plate integral):
    B0 = np.array([[-p, 0, 0, 0, 0],            # e11 = u0,x        (sinsin)
                   [0, -q, 0, 0, 0],            # e22 = v0,y        (sinsin)
                   [q, p, 0, 0, 0.0]])          # g12 = u0,y + v0,x (coscos)
    B1 = np.array([[0, 0, 0, -p, 0],            # k11 = phx,x       (sinsin)
                   [0, 0, 0, 0, -q],            # k22 = phy,y       (sinsin)
                   [0, 0, 0, q, p]], float)     # k12 = phx,y+phy,x (coscos)
    B3 = -c1 * np.array([[0, 0, -p * p, -p, 0],   # -c1(phx,x + w,xx)
                         [0, 0, -q * q, 0, -q],   # -c1(phy,y + w,yy)
                         [0, 0, 2 * p * q, q, p]])  # -c1(...+ 2 w,xy)
    G0 = np.array([[0, 0, p, 1, 0],             # g13 = phx + w,x   (cossin)
                   [0, 0, q, 0, 1.0]])          # g23 = phy + w,y   (sincos)
    G2 = -3.0 * c1 * G0                         # gamma(z) = g0 + z^2 g2
    if theory == "fsdt":
        # the OpenSG-RM constitutive law IS the MSG-homogenized 8x8: the
        # 6x6 in-plane/bending block equals CLT, but the 2x2 shear block is
        # the LS-identified section shear (core-dominated SERIES compliance,
        # G11 = 7.81e6 N/m here) -- NOT the parallel integral of G(z) dz
        # (6.85e7 N/m, 9x stiffer), and with NO shear correction factor
        r = rm_plate_msg(THICK, ANGLES, MAT_NAMES, MATERIAL_DB, fraction=0.5)
        ABDG = np.asarray(r["ABDG"])
        B06 = np.vstack([B0, B1])               # E6 rows vs d
        K = B06.T @ ABDG[:6, :6] @ B06 + G0.T @ ABDG[6:8, 6:8] @ G0
        M = np.zeros((5, 5))
        gx, gw = np.polynomial.legendre.leggauss(4)
        for lay in range(len(THICK)):
            z0, z1 = ZK[lay], ZK[lay + 1]
            zg = 0.5 * (z1 - z0) * gx + 0.5 * (z1 + z0)
            wg = 0.5 * (z1 - z0) * gw
            for z, wq in zip(zg, wg):
                mu = np.array([1, 0, 0, z, 0.0])    # FSDT: u = u0 + z phx
                mv = np.array([0, 1, 0, 0, z * 1.0])
                mw = np.array([0, 0, 1, 0, 0.0])
                M += RHO[lay] * wq * (np.outer(mu, mu) + np.outer(mv, mv)
                                      + np.outer(mw, mw))
        return K, M
    K = (B0.T @ Aab[0] @ B0                     # membrane
         + B0.T @ Aab[1] @ B1 + B1.T @ Aab[1] @ B0     # B coupling
         + B1.T @ Aab[2] @ B1                   # bending D
         + B0.T @ Aab[3] @ B3 + B3.T @ Aab[3] @ B0     # E z^3 coupling
         + B1.T @ Aab[4] @ B3 + B3.T @ Aab[4] @ B1     # F z^4 coupling
         + B3.T @ Aab[6] @ B3                   # H z^6
         + G0.T @ Ssh[0] @ G0                   # shear A44/A55
         + G0.T @ Ssh[2] @ G2 + G2.T @ Ssh[2] @ G0     # shear D z^2
         + G2.T @ Ssh[4] @ G2)                  # shear F z^4
    # consistent mass: u_dot amplitude rows m_u(z) etc., integrated with the
    # per-layer density (Gauss-4 per layer is exact for the z^6 products)
    M = np.zeros((5, 5))
    gx, gw = np.polynomial.legendre.leggauss(4)
    for lay in range(len(THICK)):
        z0, z1 = ZK[lay], ZK[lay + 1]
        zg = 0.5 * (z1 - z0) * gx + 0.5 * (z1 + z0)
        wg = 0.5 * (z1 - z0) * gw
        for z, wq in zip(zg, wg):
            mu = np.array([1, 0, -c1 * z ** 3 * p, z - c1 * z ** 3, 0.0])
            mv = np.array([0, 1, -c1 * z ** 3 * q, 0, z - c1 * z ** 3])
            mw = np.array([0, 0, 1, 0, 0.0])
            M += RHO[lay] * wq * (np.outer(mu, mu) + np.outer(mv, mv)
                                  + np.outer(mw, mw))
    return K, M


def modal_response(K, M, pulse, tgrid):
    """w_center(t): eigen-decompose the 5x5, then the CLOSED-FORM response of
    each SDOF eta_dd + om^2 eta = g*F(t) with eta(0) = eta_d(0) = 0, F(t)
    per the paper: 'step' (pure, the Abaqus decks), 'step_t1'/'sine'/'tri'
    (cut off at T1), 'blast' e^(-c t).  Piecewise closed forms; t > T1
    continues as free vibration from the exact state at T1."""
    from scipy.linalg import eigh
    lam, V = eigh(K, M)         # generalized SYMMETRIC problem: om^2, modes
    # scipy returns V with V^T M V = I -- the modal equations decouple as-is
    w = np.zeros_like(tgrid)
    for i in range(5):
        om = np.sqrt(lam[i])
        g = V[2, i] * Q0                # load enters the W equation only
        t = tgrid
        if pulse == "step":             # 1 for all t
            eta = g / om ** 2 * (1 - np.cos(om * t))
        elif pulse == "step_t1":        # 1 up to T1, then 0
            eta = g / om ** 2 * (1 - np.cos(om * t))
            late = t > T1
            eta[late] = g / om ** 2 * (np.cos(om * (t[late] - T1))
                                       - np.cos(om * t[late]))
        elif pulse == "blast":          # e^(-c t), c = CBLAST
            c = CBLAST
            den = om ** 2 + c ** 2
            eta = g / den * (np.exp(-c * t) - np.cos(om * t)
                             + (c / om) * np.sin(om * t))
        elif pulse == "sine":           # sin(pi t/T1) up to T1, then free
            Om = np.pi / T1
            den = om ** 2 - Om ** 2
            eta = g / den * (np.sin(Om * t) - (Om / om) * np.sin(om * t))
            e1 = g / den * (np.sin(Om * T1) - (Om / om) * np.sin(om * T1))
            v1 = g / den * (Om * np.cos(Om * T1) - Om * np.cos(om * T1))
            late = t > T1
            tau = t[late] - T1
            eta[late] = e1 * np.cos(om * tau) + v1 / om * np.sin(om * tau)
        elif pulse == "tri":            # 1 - t/T1 up to T1, then free
            eta = (g / om ** 2 * (1 - np.cos(om * t))
                   - g / (T1 * om ** 2) * (t - np.sin(om * t) / om))
            e1 = (g / om ** 2 * (1 - np.cos(om * T1))
                  - g / (T1 * om ** 2) * (T1 - np.sin(om * T1) / om))
            v1 = (g / om * np.sin(om * T1)
                  - g / (T1 * om ** 2) * (1 - np.cos(om * T1)))
            late = t > T1
            tau = t[late] - T1
            eta[late] = e1 * np.cos(om * tau) + v1 / om * np.sin(om * tau)
        else:
            raise ValueError(pulse)
        w += V[2, i] * eta              # w amplitude = W-row of the mode
    return w


def main():
    tgrid = np.arange(0.0, TTOT + 1e-12, DT_OUT)
    Kt, Mt = navier_KM("tsdt")
    Kf, Mf = navier_KM("fsdt")
    from scipy.linalg import eigh
    ft = np.sqrt(eigh(Kt, Mt, eigvals_only=True)) / (2 * np.pi)
    ff = np.sqrt(eigh(Kf, Mf, eigvals_only=True)) / (2 * np.pi)
    print("mode-(1,1) natural frequencies [Hz]:")
    print("  Reddy TSDT :", " ".join("%9.2f" % v for v in ft))
    print("  OpenSG-RM  :", " ".join("%9.2f" % v for v in ff))
    for kind in ("step", "blast"):
        wt = modal_response(Kt, Mt, kind, tgrid)
        wf = modal_response(Kf, Mf, kind, tgrid)
        np.savetxt(os.path.join(HERE, "reddy_wt_%s.dat" % kind),
                   np.column_stack([tgrid, wt, wf]),
                   header="t[s]  w_center_TSDT[m]  w_center_RM_navier[m]  "
                          "(Nayak Ex.5, %s pulse, q0 F(t) sinsin)" % kind)
        ipk_t, ipk_f = np.argmax(np.abs(wt)), np.argmax(np.abs(wf))
        print("%6s: peak w TSDT %.5e m @ %.3f ms | RM-Navier %.5e m @ %.3f"
              " ms (%+.2f %%)" % (kind, wt[ipk_t], 1e3 * tgrid[ipk_t],
                                  wf[ipk_f], 1e3 * tgrid[ipk_f],
                                  100 * (wf[ipk_f] - wt[ipk_t]) / wt[ipk_t]))
    # the Nayak Fig.-13 reproduction: all four pulses, his w/0.0254 scaling
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for pulse, lab, sty in (("sine", "sine", "-"), ("step_t1", "step", "--"),
                            ("tri", "triangular", "-."),
                            ("blast", "blast", ":")):
        wt = modal_response(Kt, Mt, pulse, tgrid)
        ax.plot(1e3 * tgrid, wt / 0.0254, sty, lw=1.6, label=lab)
    ax.set_xlabel("time [ms]", fontsize=11)
    ax.set_ylabel(r"$w_{\rm center}/0.0254$ m  (Nayak's scaling)",
                  fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reddy_fig13.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote reddy_wt_step/blast.dat + reddy_fig13.png")
    # ---- Ex.2 anchor: pin THIS TSDT to Nayak's own tabulated numbers -----
    set_case("ex2")
    K2, M2 = navier_KM("tsdt")
    K2f, M2f = navier_KM("fsdt")
    tchk = np.array([1.6e-3, 3.2e-3, 4.8e-3, 6.4e-3, 8.0e-3])
    w2 = modal_response(K2, M2, "step_t1", tchk) / 0.0254
    w2f = modal_response(K2f, M2f, "step_t1", tchk) / 0.0254
    # Nayak Table 3 (the 9-NODE element, 4x4 mesh, dt = 40 us -- his most
    # converged printed values; the 4-node Table 2 is still shear-locked)
    nay = [0.1870, 0.6180, 0.9983, 0.4548, -0.2191]
    print("\nEx.2 anchor (0/90/0, a=5h, step t1=6ms): w/0.0254 m")
    print("  t [ms]        :", " ".join("%8.1f" % (1e3 * t) for t in tchk))
    print("  Nayak Tbl3 P9 :", " ".join("%8.4f" % v for v in nay))
    print("  TSDT Navier   :", " ".join("%8.4f" % v for v in w2))
    print("  OpenSG-RM Nav.:", " ".join("%8.4f" % v for v in w2f))
    set_case("ex5")


if __name__ == "__main__":
    main()
