"""models.py -- the plate-level models compared against exact 3-D elasticity.

All are driven by the SAME load sigma33(x, top) = q0 sin(px) as ``exact_cyl.ExactCyl``.
Because cylindrical bending is statically determinate at the plate level,

    N11 = N12 = 0,   M11 = q0/p^2,   M12 = 0,   Q1 = q0/p,   Q2 = 0,

every model that uses the same ABD sees identical stress resultants.  The comparison is
therefore purely about how each DISTRIBUTES them through the thickness.

  fsdt          single director + shear-correction factor k = 5/6.  sigma13 is piecewise
                CONSTANT, discontinuous at every interface, nonzero on the traction-free
                faces; sigma33 is not predictable at all.  This is the baseline that
                Garg et al. (2023) patch with a Gaussian-process surrogate.
  clt_equil     classical equilibrium ("shear flow") recovery -- integrate sigma11,1
                through the thickness using the CLT sigma11 (Whitney 1973).
  msg           MSG-VAM: first-order gradient warping supplies sigma13 and sigma23
                directly; sigma33 then follows from through-thickness equilibrium.
"""
import numpy as np

from jaxcfg import jax, jnp
import sg_plate as SG

IX_CYL = np.array([0, 2, 3, 5])      # active plate strains in cylindrical bending


# ------------------------------------------------------------------ plate strains
def plate_strains(A6, p, q0=1.0):
    """Plate-strain amplitude Ehat (6,) with eps22 = kappa22 = 0 (infinite in y)."""
    A6 = jnp.asarray(A6)
    Ar = A6[jnp.ix_(jnp.asarray(IX_CYL), jnp.asarray(IX_CYL))]
    F = jnp.array([0.0, 0.0, q0 / p ** 2, 0.0])          # [N11, N12, M11, M12]
    Er = jnp.linalg.solve(Ar, F)
    return jnp.zeros(6).at[jnp.asarray(IX_CYL)].set(Er)


def plane_stress_C(C):
    """Classical 3x3 Q from the 6x6 (sigma33 = sigma13 = sigma23 = 0)."""
    keep = jnp.asarray([0, 1, 5]); drop = jnp.asarray([2, 3, 4])
    return (C[jnp.ix_(keep, keep)]
            - C[jnp.ix_(keep, drop)] @ jnp.linalg.solve(C[jnp.ix_(drop, drop)],
                                                        C[jnp.ix_(drop, keep)]))


# -------------------------------------------------------------------------- FSDT
def fsdt_shear(sg, ks=5.0 / 6.0):
    """FSDT transverse-shear stiffness (single scalar correction factor) and the
    per-ply 2x2 blocks mapping [g13, g23] -> [s13, s23]."""
    C = sg['C_layers']
    gk = jnp.stack([jnp.array([[c[4, 4], c[4, 3]], [c[3, 4], c[3, 3]]]) for c in C])
    Gs = ks * jnp.einsum('k,kab->ab', jnp.asarray(sg['mesh']['thick']), gk)
    return Gs, gk


def fsdt(sg, E6, p, q0=1.0, n_per_layer_out=41, ks=5.0 / 6.0):
    """FSDT profile.  ``s33`` is None -- the theory cannot produce it."""
    m = sg['mesh']
    z, e, _ = SG.sample_points(m, n_per_layer_out)
    lay = m['elem_layer'][e]
    Gs, gk = fsdt_shear(sg, ks)
    gam = jnp.linalg.solve(Gs, jnp.array([q0 / p, 0.0]))
    sv = jnp.einsum('nab,b->na', gk[jnp.asarray(lay)], gam)
    s11 = _clt_inplane(sg, E6, z, lay)[:, 0]
    return {'z': z, 's11': np.asarray(s11), 's13': np.asarray(sv[:, 0]),
            's23': np.asarray(sv[:, 1]), 's33': None, 'gamma': np.asarray(gam)}


def _clt_inplane(sg, E6, z, lay):
    """Classical sigma_{11,22,12} at depths ``z`` (already referred to z_ref)."""
    Cr = jax.vmap(plane_stress_C)(sg['C_layers'])[jnp.asarray(lay)]
    zz = jnp.asarray(z)
    E6 = jnp.asarray(E6)
    g = jnp.stack([E6[0] + zz * E6[3], E6[1] + zz * E6[4], E6[2] + zz * E6[5]], axis=1)
    return jnp.einsum('nab,nb->na', Cr, g)


# ------------------------------------------------------- classical equilibrium
def clt_equil(sg, E6, p, n_per_layer_out=41, n_gauss=10):
    """sigma13' = -p sigma11_CLT integrated from the bottom face; then sigma33."""
    m = sg['mesh']
    z, e, _ = SG.sample_points(m, n_per_layer_out)
    lay = m['elem_layer'][e]
    bot = float(-m['z_ref'])

    xi, wq = np.polynomial.legendre.leggauss(n_gauss)
    zprev = np.concatenate([[bot], z[:-1]])
    a = zprev[:, None]; b = z[:, None]
    zg = 0.5 * (a + b) + 0.5 * (b - a) * xi[None, :]
    wt = 0.5 * (b - a) * wq[None, :]
    lay_g = np.clip(np.searchsorted(
        np.cumsum(m['thick']) - m['z_ref'], zg.ravel(), side='left'),
        0, m['thick'].size - 1)
    s11g = _clt_inplane(sg, E6, zg.ravel(), lay_g)[:, 0].reshape(zg.shape)
    s12g = _clt_inplane(sg, E6, zg.ravel(), lay_g)[:, 2].reshape(zg.shape)
    s13 = jnp.cumsum(-p * jnp.sum(jnp.asarray(wt) * s11g, axis=1))
    s23 = jnp.cumsum(-p * jnp.sum(jnp.asarray(wt) * s12g, axis=1))

    ip = _clt_inplane(sg, E6, z, lay)
    s33 = sigma33_equilibrium(z, s13, p1=p)
    return {'z': z, 's11': np.asarray(ip[:, 0]), 's13': np.asarray(s13),
            's23': np.asarray(s23), 's33': np.asarray(s33)}


# ------------------------------------------------------------------------- MSG
def msg(sg, E6, p, n_per_layer_out=41):
    """MSG-VAM recovery.

    In-plane stresses come from the zeroth-order warping at mid-span (where the Navier
    gradient vanishes); the transverse shear comes from the FIRST-ORDER gradient warping
    at the support, driven by dE/dx1 = p*Ehat; sigma33 then follows from through-thickness
    equilibrium of that recovered sigma13.
    """
    E6 = jnp.asarray(E6)
    dE1 = p * E6
    z, _, Sig_m = SG.recover(sg, E6, None, None, n_per_layer_out)          # mid-span
    _, _, Sig_s = SG.recover(sg, jnp.zeros(6), dE1, None, n_per_layer_out)  # support
    s13 = Sig_s[:, 4]
    s23 = Sig_s[:, 3]
    s33 = sigma33_equilibrium(z, s13, p1=p)
    return {'z': z, 's11': np.asarray(Sig_m[:, 0]), 's22': np.asarray(Sig_m[:, 1]),
            's12': np.asarray(Sig_m[:, 5]), 's13': np.asarray(s13),
            's23': np.asarray(s23), 's33': np.asarray(s33),
            's33_direct': np.asarray(Sig_m[:, 2])}


def sigma33_equilibrium(z, s13, s23=None, p1=1.0, p2=0.0):
    """sigma33 from  dsigma33/dz = p1*sigma13 (+ p2*sigma23), integrated from the bottom.

    The top-face value comes out as q0 automatically whenever the recovered sigma13
    integrates to Q1 -- which both the classical and the MSG recovery guarantee.
    """
    z = jnp.asarray(z)
    f = p1 * jnp.asarray(s13)
    if s23 is not None:
        f = f + p2 * jnp.asarray(s23)
    trap = 0.5 * (f[1:] + f[:-1]) * jnp.diff(z)
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(trap)])


# ------------------------------------------------------------------------ driver
def run(thick, angles, mats, matdb, S, q0=1.0, n_per_layer_out=61, npl_sg=6, order=3):
    """Solve all four models for one laminate at slenderness S = L/h."""
    from exact_cyl import ExactCyl
    thick = np.asarray(thick, float)
    h = float(thick.sum())
    ex = ExactCyl(thick, angles, mats, matdb, S * h, q0=q0)
    sg = SG.build(thick, angles, mats, matdb, n_per_layer=npl_sg, elem_order=order)
    E6 = plate_strains(sg['A6'], ex.p, q0=q0)
    zc, sig_e, eps_e, uvw_e = ex.profile(n_per_layer=n_per_layer_out)
    out = dict(S=S, h=h, p=ex.p, zc=zc, exact=sig_e, sg=sg, E6=np.asarray(E6),
               fsdt=fsdt(sg, E6, ex.p, q0, n_per_layer_out),
               clt=clt_equil(sg, E6, ex.p, n_per_layer_out),
               msg=msg(sg, E6, ex.p, n_per_layer_out),
               thick=thick)
    assert np.allclose(zc, out['fsdt']['z'], atol=1e-9 * h), "z grids disagree"
    return out


def relerr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))
