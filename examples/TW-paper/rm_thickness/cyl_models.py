"""cyl_models.py -- the three plate-level models for cylindrical bending, all driven by
the SAME sinusoidal load sigma33(x, top) = q0 sin(p x) as ``exact_cyl.ExactCyl``.

Because the problem is statically determinate at the plate level,

    N11 = N12 = 0,     M11 = q0 / p^2,     M12 = 0,     Q1 = q0 / p,     Q2 = 0,

so every model that uses the same ABD sees the SAME in-plane strains.  The models differ
ONLY in how they recover the through-thickness distributions:

  fsdt_profile()      constitutive shear from a single director + shear-correction factor
                      -> sigma13 piecewise CONSTANT, discontinuous, nonzero on the faces;
                      sigma33 not available at all.  (= the baseline in Garg et al. 2023)
  clt_equil_profile() classical equilibrium recovery: integrate sigma11,1 through the
                      thickness (Whitney 1973 shear flow) using the CLT sigma11.
  msg_profile()       MSG-VAM: first-order (gradient) warping supplies sigma13 directly,
                      then sigma33 follows from through-thickness equilibrium.

Amplitudes returned are for the sin/cos split:
  sigma_{11,22,33,12}  multiply sin(p x)   (max at mid-span)
  sigma_{13,23}        multiply cos(p x)   (max at the supports)
"""
import numpy as np

from msg_rm_plate import (rm_plate_msg, msgrm_recover_profile, sigma33_equilibrium,
                          _elem_of)

IX_CYL = np.array([0, 2, 3, 5])       # active plate strains: e11, g12, k11, k12


def plate_strains(A6, p, q0=1.0):
    """Cylindrical-bending plate strain amplitude Ehat (6,), with eps22 = kappa22 = 0."""
    Ar = A6[np.ix_(IX_CYL, IX_CYL)]
    F = np.array([0.0, 0.0, q0 / p ** 2, 0.0])       # [N11, N12, M11, M12]
    Er = np.linalg.solve(Ar, F)
    E6 = np.zeros(6)
    E6[IX_CYL] = Er
    return E6


def _plane_stress_C(C):
    """Reduce the 6x6 to the classical 3x3 Q (sigma33 = sigma13 = sigma23 = 0)."""
    keep = np.array([0, 1, 5]); drop = np.array([2, 3, 4])
    return (C[np.ix_(keep, keep)]
            - C[np.ix_(keep, drop)] @ np.linalg.solve(C[np.ix_(drop, drop)],
                                                      C[np.ix_(drop, keep)]))


def _sample_z(obj, n_per_layer, eps=1e-9):
    thick = obj['thick']
    bot = np.concatenate([[0.0], np.cumsum(thick)]) - obj['z_ref']
    out = []
    for k in range(len(thick)):
        a, b = bot[k], bot[k + 1]
        out.append(np.linspace(a + eps * (b - a), b - eps * (b - a), n_per_layer))
    return np.concatenate(out)


# --------------------------------------------------------------------- CLT / FSDT
def clt_inplane(obj, E6, z):
    """Classical (plane-stress) sigma_{11,22,12} amplitude at depth z."""
    e = _elem_of(obj, z)
    k = obj['elem_layer'][e]
    Cr = _plane_stress_C(obj['C_layers'][k])
    g = np.array([E6[0] + z * E6[3], E6[1] + z * E6[4], E6[2] + z * E6[5]])
    return Cr @ g


def fsdt_shear_stiffness(obj, ks=5.0 / 6.0):
    """FSDT [A44,A45;A45,A55]-style 2x2 with a single scalar correction factor.

    Uses the same plane-stress-consistent transverse moduli as the SG (C[3:5,3:5]).
    Returns (Gs_plate, list of per-layer 2x2 transverse blocks).
    """
    thick = obj['thick']
    blocks = []
    Gs = np.zeros((2, 2))
    for k in range(len(thick)):
        C = obj['C_layers'][k]
        gk = np.array([[C[4, 4], C[4, 3]], [C[3, 4], C[3, 3]]])   # [s13;s23] = gk [g13;g23]
        blocks.append(gk)
        Gs += gk * thick[k]
    return ks * Gs, blocks


def fsdt_profile(obj, E6, p, q0=1.0, n_per_layer=41, ks=5.0 / 6.0):
    """FSDT through-thickness profile.  sigma33 is returned as None (not predictable)."""
    z = _sample_z(obj, n_per_layer)
    Gs, blocks = fsdt_shear_stiffness(obj, ks=ks)
    gam = np.linalg.solve(Gs, np.array([q0 / p, 0.0]))            # [2g13, 2g23]
    s11 = np.empty(z.size); s13 = np.empty(z.size); s23 = np.empty(z.size)
    for i, zz in enumerate(z):
        e = _elem_of(obj, zz)
        k = obj['elem_layer'][e]
        s11[i] = clt_inplane(obj, E6, zz)[0]
        sv = blocks[k] @ gam
        s13[i] = sv[0]; s23[i] = sv[1]
    return {'z': z, 's11': s11, 's13': s13, 's23': s23, 's33': None, 'gamma': gam}


def clt_equil_profile(obj, E6, p, n_per_layer=41, n_gauss=8):
    """Classical equilibrium ("shear flow") recovery: sigma13' = -p * sigma11_CLT."""
    z = _sample_z(obj, n_per_layer)
    thick = obj['thick']
    bot = np.concatenate([[0.0], np.cumsum(thick)]) - obj['z_ref']

    # exact per-element integral of the (piecewise linear in z) CLT sigma11
    xi, wq = np.polynomial.legendre.leggauss(n_gauss)
    s13 = np.zeros(z.size)
    acc = 0.0
    zprev = bot[0]
    for i, zz in enumerate(z):
        a, b = zprev, zz
        if b > a:
            zg = 0.5 * (a + b) + 0.5 * (b - a) * xi
            wt = 0.5 * (b - a) * wq
            acc += -p * sum(w * clt_inplane(obj, E6, zc)[0] for zc, w in zip(zg, wt))
        s13[i] = acc
        zprev = zz
    s11 = np.array([clt_inplane(obj, E6, zz)[0] for zz in z])
    s33 = sigma33_equilibrium(z, s13, p1=p)
    return {'z': z, 's11': s11, 's13': s13, 's23': np.zeros_like(s13), 's33': s33}


# ------------------------------------------------------------------------- MSG-VAM
def msg_profile(obj, E6, p, n_per_layer=41):
    """MSG-VAM recovery.

    In-plane stresses come from the zeroth-order (classical) warping evaluated with the
    plate strains E; the transverse shear comes from the FIRST-ORDER gradient warping
    driven by dE/dx1 = p * E6 (the Navier gradient); sigma33 then follows from
    through-thickness equilibrium of the recovered sigma13.
    """
    dE1 = p * E6
    # in-plane station (mid-span): E = Ehat, dE1 = 0
    z, _, Sig_m = msgrm_recover_profile(obj, E6, None, None, n_per_layer=n_per_layer)
    # transverse-shear station (support): E = 0, dE1 = p*Ehat
    _, _, Sig_s = msgrm_recover_profile(obj, np.zeros(6), dE1, None,
                                        n_per_layer=n_per_layer)
    s13 = Sig_s[:, 4]
    s23 = Sig_s[:, 3]
    s33 = sigma33_equilibrium(z, s13, p1=p)
    return {'z': z, 's11': Sig_m[:, 0], 's22': Sig_m[:, 1], 's12': Sig_m[:, 5],
            's13': s13, 's23': s23, 's33': s33,
            's33_direct': Sig_m[:, 2]}


def build(thick, angles, mats, matdb, n_per_layer=4, elem_order=3):
    h = float(np.sum(thick))
    return rm_plate_msg(thick, angles, mats, matdb, n_per_layer=n_per_layer,
                        elem_order=elem_order, z_ref=h / 2.0)
