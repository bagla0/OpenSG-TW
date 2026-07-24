"""batched.py -- fully-traced versions of the SG recovery and the exact solution, so a
whole POPULATION of laminates can be pushed through ``jax.vmap`` in one call.

The trick that makes this possible: sample the through-thickness profile at fixed
FRACTIONS within each ply.  The element index and the local coordinate xi of a sample
point then depend only on that fraction and on the (static) mesh structure -- never on the
actual ply thicknesses.  So the whole point-location problem is host-side and static, and
everything that touches the traced ``thick`` and ``C_layers`` is pure JAX.
"""
import numpy as np
from jax.scipy.linalg import expm

from jaxcfg import jax, jnp
from materials import shape_tensors, basis_at
from sg_plate import sg_plate_solve
from exact_cyl import _layer_A


def sample_spec(nlay, n_per_layer, p, n_out, eps=1e-9):
    """Host-side sampling plan: (lay, elem, frac, N, dN, idx) -- all static."""
    frac = np.linspace(eps, 1.0 - eps, n_out)
    lay = np.repeat(np.arange(nlay), n_out)
    fr = np.tile(frac, nlay)
    loc = np.clip((fr * n_per_layer).astype(int), 0, n_per_layer - 1)
    elem = lay * n_per_layer + loc
    xi = 2.0 * (fr * n_per_layer - loc) - 1.0
    N, dN = basis_at(p, xi)
    idx = jnp.asarray(3 * p * elem[:, None] + np.arange(3 * (p + 1))[None, :])
    return dict(lay=jnp.asarray(lay), elem=jnp.asarray(elem), fr=jnp.asarray(fr),
                N=N, dN=dN, idx=idx, n_per_layer=n_per_layer, p=p,
                nlay=nlay, n_out=n_out)


def sg_recover(sol, thick, z_ref, spec, E6, dE1, dE2):
    """Traced through-thickness recovery.  Returns (z, Sig)."""
    p = spec['p']
    E_B, E_M1, E_M2, GE0, GE1 = shape_tensors(p)
    bot = jnp.concatenate([jnp.zeros(1), jnp.cumsum(thick)])[:-1] - z_ref
    lay = spec['lay']
    z = bot[lay] + spec['fr'] * thick[lay]
    he = thick[lay] / spec['n_per_layer']

    idx = spec['idx']
    V0e = sol['V0'][idx]
    C1e = sol['C1bar'][idx]
    C2e = sol['C2bar'][idx]
    w_loc = V0e @ E6 + C1e @ dE1 + C2e @ dE2
    g1 = V0e @ dE1
    g2 = V0e @ dE2

    B = jnp.einsum('nj,jab->nab', spec['dN'] * (2.0 / he)[:, None], E_B)
    M1 = jnp.einsum('nj,jab->nab', spec['N'], E_M1)
    M2 = jnp.einsum('nj,jab->nab', spec['N'], E_M2)
    Ge = GE0[None] + z[:, None, None] * GE1[None]

    Gam = (jnp.einsum('nab,nb->na', B, w_loc)
           + jnp.einsum('nab,b->na', Ge, E6)
           + jnp.einsum('nab,nb->na', M1, g1)
           + jnp.einsum('nab,nb->na', M2, g2))
    return z, jnp.einsum('nab,nb->na', sol['C_layers'][lay], Gam)


def exact_profile(thick, C_layers, S, spec, q0=1.0):
    """Traced exact 3-D elasticity profile at the same sample points."""
    h = jnp.sum(thick)
    p = jnp.pi / (S * h)
    pt = p * h
    E0 = jnp.max(jnp.abs(C_layers))
    t_hat = thick / h
    A = jax.vmap(_layer_A, in_axes=(0, None, None))(C_layers, pt, E0)
    Tk = jax.vmap(lambda a, t: expm(a * t))(A, t_hat)
    T_tot, T_cum = jax.lax.scan(lambda T, Ti: (Ti @ T, T), jnp.eye(6), Tk)
    uvw0 = jnp.linalg.solve(T_tot[3:6, 0:3], jnp.array([0.0, 0.0, q0 / E0]))
    s0 = jnp.concatenate([uvw0, jnp.zeros(3)])

    zb = jnp.concatenate([jnp.zeros(1), jnp.cumsum(t_hat)])[:-1]
    lay = spec['lay']
    z_hat = zb[lay] + spec['fr'] * t_hat[lay]

    def one(zi, ki):
        s = expm(A[ki] * (zi - zb[ki])) @ (T_cum[ki] @ s0)
        U, V, W, X, Y, Z = s
        C = C_layers[ki]
        C11, C12, C13, C16 = C[0, 0], C[0, 1], C[0, 2], C[0, 5]
        C23, C26 = C[1, 2], C[1, 5]
        C33, C36, C66 = C[2, 2], C[2, 5], C[5, 5]
        e33 = (pt * C13 * U + pt * C36 * V + E0 * Z) / C33
        S11 = -pt * C11 * U - pt * C16 * V + C13 * e33
        S22 = -pt * C12 * U - pt * C26 * V + C23 * e33
        S12 = -pt * C16 * U - pt * C66 * V + C36 * e33
        return jnp.array([S11, S22, E0 * Z, E0 * Y, E0 * X, S12])

    # oracle integrity: the traction BCs must hold to round-off for EVERY sample,
    # otherwise a badly-scaled expm would silently poison the statistics
    st = T_tot @ s0
    bc = jnp.max(jnp.abs(jnp.array([s0[3], s0[4], s0[5],
                                    st[3], st[4], st[5] - q0 / E0]))) / (q0 / E0)
    return jax.vmap(one)(z_hat, lay), p, h, bc
