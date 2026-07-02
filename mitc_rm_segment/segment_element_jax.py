"""
segment_element_jax.py    [ Windows opensg_2_0_env ]
========================================================================
JAX port of the 2-D MITC-RM surface-quad shell element (segment_element.py),
for the tapered-segment EB path.  jit + vmap over quads, scatter assembly.

Ports EXACTLY (verified against the NumPy _quad_ops / assemble_segment):
  * _quad_ops_j : Gamma_h(BDq 6x20) / shear(BGq 2x20) / Gamma_l(BLq 6x20) + geo,
    in-frame derivatives via the metric transpose (G^T)^-1.
  * _mitc_shear_j : Dvorkin-Bathe MITC4 tying (g23 @ (0,+-1), g13 @ (+-1,0)).
  * _macro_BD_j : macro map (== msg_rm._macro_BD, KAPPA12_MACRO=-2).
  * assemble_segment_jax : per-element (kdd,kde,kee) vmapped, scattered to global.

Boundary V0 (4 EB modes) still comes from the validated 1-D RM solve; only the
segment assembly + Dirichlet EB is ported here.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

KAPPA12_MACRO = -2.0
_GP = 1.0 / np.sqrt(3.0)
QPTS = jnp.array([(-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)])
_TIE23 = jnp.array([(0.0, -1.0), (0.0, 1.0)])
_TIE13 = jnp.array([(-1.0, 0.0), (1.0, 0.0)])
_W1 = 5 * jnp.arange(4) + 0
_W2 = 5 * jnp.arange(4) + 1
_W3 = 5 * jnp.arange(4) + 2
_O1 = 5 * jnp.arange(4) + 3
_O2 = 5 * jnp.arange(4) + 4


def _bilinear_j(xi, eta):
    xc = jnp.array([-1., 1., 1., -1.]); ec = jnp.array([-1., -1., 1., 1.])
    N = 0.25 * (1 + xc * xi) * (1 + ec * eta)
    dNx = 0.25 * xc * (1 + ec * eta)
    dNe = 0.25 * ec * (1 + xc * xi)
    return N, dNx, dNe


@partial(jax.jit, static_argnums=(7,))
def _quad_ops_j(X, e1, e2, e3, xi, eta, k22, cross):
    N, dNx, dNe = _bilinear_j(xi, eta)
    Jxi = dNx @ X; Jeta = dNe @ X
    G = jnp.array([[e1 @ Jxi, e1 @ Jeta], [e2 @ Jxi, e2 @ Jeta]])
    d = (jnp.linalg.inv(G.T) @ jnp.vstack([dNx, dNe])).T        # (4,2): [d/dx1, d/ds]
    dA = jnp.linalg.norm(jnp.cross(Jxi, Jeta))
    x = N @ X
    c0, c1 = cross
    x2, x3 = x[c0], x[c1]
    t2, t3 = e2[c0], e2[c1]
    n2, n3 = t3, -t2
    D1, Ds, Na = d[:, 0], d[:, 1], N

    BDq = jnp.zeros((6, 20))
    BDq = BDq.at[0, _W1].add(D1)
    BDq = BDq.at[1, _W2].add(t2 * Ds).at[1, _W3].add(t3 * Ds)
    BDq = BDq.at[2, _W1].add(Ds).at[2, _W2].add(t2 * D1).at[2, _W3].add(t3 * D1)
    BDq = BDq.at[3, _O2].add(D1)
    BDq = BDq.at[4, _O1].add(Ds)
    BDq = BDq.at[5, _O2].add(Ds).at[5, _W1].add(0.5 * k22 * Ds).at[5, _O1].add(D1)

    BGq = jnp.zeros((2, 20))
    BGq = BGq.at[0, _O2].add(Na).at[0, _W2].add(n2 * D1).at[0, _W3].add(n3 * D1)
    BGq = BGq.at[1, _W2].add(n2 * Ds).at[1, _W3].add(n3 * Ds).at[1, _O1].add(-Na)

    BLq = jnp.zeros((6, 20))
    BLq = BLq.at[0, _W1].add(Na)
    BLq = BLq.at[2, _W2].add(t2 * Na).at[2, _W3].add(t3 * Na)
    BLq = BLq.at[5, _W2].add(2 * t3 * Ds - 0.5 * k22 * t2 * Na)
    BLq = BLq.at[5, _W3].add(-2 * t2 * Ds - 0.5 * k22 * t3 * Na)
    return BDq, BGq, BLq, (x2, x3, t2, t3, dA)


@partial(jax.jit, static_argnums=(7,))
def _mitc_shear_j(X, e1, e2, e3, xi, eta, k22, cross):
    A23 = _quad_ops_j(X, e1, e2, e3, _TIE23[0, 0], _TIE23[0, 1], k22, cross)[1][1]
    B23 = _quad_ops_j(X, e1, e2, e3, _TIE23[1, 0], _TIE23[1, 1], k22, cross)[1][1]
    A13 = _quad_ops_j(X, e1, e2, e3, _TIE13[0, 0], _TIE13[0, 1], k22, cross)[1][0]
    B13 = _quad_ops_j(X, e1, e2, e3, _TIE13[1, 0], _TIE13[1, 1], k22, cross)[1][0]
    g23 = 0.5 * (1.0 - eta) * A23 + 0.5 * (1.0 + eta) * B23
    g13 = 0.5 * (1.0 - xi) * A13 + 0.5 * (1.0 + xi) * B13
    return jnp.vstack([g13, g23])


def _macro_BD_j(x2, x3, t2, t3, k22):
    Rn = x2 * t3 - x3 * t2
    B = jnp.zeros((6, 4))
    B = B.at[0].set(jnp.array([1.0, 0.0, x3, -x2]))
    B = B.at[2, 1].set(Rn)
    B = B.at[3, 2].set(t2).at[3, 3].set(t3)
    B = B.at[5, 1].set(KAPPA12_MACRO + 0.5 * k22 * Rn)
    return B


@partial(jax.jit, static_argnums=(4,))
def _elem_matrices(X, e1, e2, e3, cross, D, G, k22):
    kdd = jnp.zeros((20, 20)); kde = jnp.zeros((20, 4)); kee = jnp.zeros((4, 4))
    for q in range(4):
        xi, eta = QPTS[q, 0], QPTS[q, 1]
        BDq, BGq, BLq, geo = _quad_ops_j(X, e1, e2, e3, xi, eta, k22, cross)
        x2, x3, t2, t3, dA = geo
        BGb = _mitc_shear_j(X, e1, e2, e3, xi, eta, k22, cross)
        BDe = _macro_BD_j(x2, x3, t2, t3, k22)
        kdd = kdd + (BDq.T @ D @ BDq + BGb.T @ G @ BGb) * dA
        kde = kde + (BDq.T @ D @ BDe) * dA
        kee = kee + (BDe.T @ D @ BDe) * dA
    return kdd, kde, kee


def assemble_segment_jax(nodes, quads, subdom, e1s, e2s, e3s, D_stack, G_stack, k22_e, cross=(1, 2)):
    """Assemble Dhh (ndof,ndof), Dhe (ndof,4), Dee (4,4).  D_stack (nsec,6,6),
    G_stack (nsec,2,2); k22_e (Ne,).  vmap the per-element matrices, scatter."""
    nodes = jnp.asarray(nodes); quads = jnp.asarray(quads)
    e1s = jnp.asarray(e1s); e2s = jnp.asarray(e2s); e3s = jnp.asarray(e3s)
    Ne = quads.shape[0]; Nn = nodes.shape[0]; ndof = 5 * Nn
    Xe = nodes[quads]                                           # (Ne,4,3)
    De = jnp.asarray(D_stack)[jnp.asarray(subdom)]              # (Ne,6,6)
    Ge = jnp.asarray(G_stack)[jnp.asarray(subdom)]              # (Ne,2,2)
    k22 = jnp.asarray(k22_e)
    vm = jax.vmap(lambda X, a, b, c, D, G, k: _elem_matrices(X, a, b, c, cross, D, G, k))
    kdd, kde, kee = vm(Xe, e1s, e2s, e3s, De, Ge, k22)          # (Ne,20,20),(Ne,20,4),(Ne,4,4)
    gi = (5 * quads[:, :, None] + jnp.arange(5)[None, None, :]).reshape(Ne, 20)   # (Ne,20)
    ri = jnp.broadcast_to(gi[:, :, None], (Ne, 20, 20)).reshape(-1)
    ci = jnp.broadcast_to(gi[:, None, :], (Ne, 20, 20)).reshape(-1)
    Dhh = jnp.zeros((ndof, ndof)).at[ri, ci].add(kdd.reshape(-1))
    Dhe = jnp.zeros((ndof, 4)).at[gi.reshape(-1)].add(kde.reshape(-1, 4))
    Dee = kee.sum(0)
    return Dhh, Dhe, Dee
