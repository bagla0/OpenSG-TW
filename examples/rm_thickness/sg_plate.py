"""sg_plate.py -- the MSG 1-D Structure-Gene plate model, in JAX.

This is the FEA of the study: a finite element mesh through the laminate THICKNESS.
p-th order Lagrange elements, three warping degrees of freedom per node.  A three-ply
laminate with four elements per ply and p = 3 has 111 DOF; the whole model solves in
microseconds once jitted, and ``jax.vmap`` maps it over a whole population of laminates.

Construction (Yu, Hodges & Volovoi, Int. J. Solids Struct. 39 (2002) 5185; Yu, Int. J.
Solids Struct. 42 (2005) 6680):

  zeroth order   V0                  ->  A6, the classical ABD
  first order    C1bar, C2bar        ->  warping driven by the in-plane strain gradients
  second order   H (12x12)           ->  the residual energy U*
  RM projection  least squares of U* over X = G^{-1} and the relaxed constants
                                     ->  G_msg, the MSG transverse-shear stiffness

Voigt order [11,22,33,23,13,12]; plate strain E = [e11,e22,g12,k11,k22,k12]; transverse
shear gamma = [2g13, 2g23].  The thickness coordinate runs from the BOTTOM face minus
``z_ref``.

Ported from ``examples/TW-paper/xsec_paper/msg_rm_plate.py`` (numpy); the port is pinned
to its parent by ``tests/test_against_numpy.py``.
"""
from functools import partial

import numpy as np

from jaxcfg import jax, jnp
from materials import shape_tensors, quadrature, basis_at, layer_stiffness

N_LS = 3 + 36          # RM-projection unknowns: sym X (3) + relaxed constants (36)


# ------------------------------------------------------------------------ mesh
def sg_mesh(thick, n_per_layer, p, z_ref):
    """Host-side mesh: everything whose SHAPE is fixed by the layup."""
    thick = np.asarray(thick, float)
    nlay = thick.size
    elem_layer = np.repeat(np.arange(nlay), n_per_layer)
    he = thick[elem_layer] / n_per_layer
    xl = np.concatenate([[0.0], np.cumsum(he)])[:-1] - z_ref
    n_elem = elem_layer.size
    idx = 3 * p * np.arange(n_elem)[:, None] + np.arange(3 * (p + 1))[None, :]
    nodes_e = xl[:, None] + he[:, None] * np.arange(p + 1)[None, :] / p
    node_x = np.concatenate([nodes_e[:, :p].ravel(), [nodes_e[-1, p]]])
    return dict(elem_layer=elem_layer, he=he, xl=xl, idx=idx, node_x=node_x,
                n_elem=n_elem, n_node=node_x.size, ndofs=3 * node_x.size,
                thick=thick, z_ref=float(z_ref), p=p, n_per_layer=n_per_layer)


def _geom(thick, n_per_layer, z_ref):
    """Traced element geometry (he, xl) from a traced ``thick``."""
    he = jnp.repeat(thick, n_per_layer) / n_per_layer
    xl = jnp.concatenate([jnp.zeros(1), jnp.cumsum(he)])[:-1] - z_ref
    return he, xl


# -------------------------------------------------------------------- assembly
def _elem_blocks(C, he, xl, xi, wq, Nq, dNq, E_B, E_M1, E_M2, GE0, GE1):
    """Every element integral for ONE element, summed over quadrature."""
    def one(q):
        x_q = xl + 0.5 * he * (1.0 + xi[q])
        dw = 0.5 * he * wq[q]
        B = jnp.einsum('i,iab->ab', dNq[q] * (2.0 / he), E_B)
        M1 = jnp.einsum('i,iab->ab', Nq[q], E_M1)
        M2 = jnp.einsum('i,iab->ab', Nq[q], E_M2)
        Ge = GE0 + x_q * GE1
        CB = C @ B; CM1 = C @ M1; CM2 = C @ M2; CGe = C @ Ge
        return (dw * (B.T @ CB), dw * (B.T @ CGe), dw * (Ge.T @ CGe),
                dw * (B.T @ CM1), dw * (B.T @ CM2),
                dw * (M1.T @ CB), dw * (M2.T @ CB),
                dw * (M1.T @ CM1), dw * (M1.T @ CM2), dw * (M2.T @ CM2),
                dw * (M1.T @ CGe), dw * (M2.T @ CGe))

    out = jax.vmap(one)(jnp.arange(xi.size))
    return tuple(jnp.sum(o, axis=0) for o in out)


@partial(jax.jit, static_argnums=(2, 3))
def sg_plate_solve(thick, C_layers, n_per_layer, p, z_ref):
    """Core MSG solve.  ``thick`` (nlay,) and ``C_layers`` (nlay,6,6) are traced; the
    element count per ply and the polynomial order are static."""
    nlay = thick.shape[0]
    n_elem = nlay * n_per_layer
    ndofs = 3 * (p * n_elem + 1)

    elem_layer = jnp.asarray(np.repeat(np.arange(nlay), n_per_layer))
    idx = jnp.asarray(3 * p * np.arange(n_elem)[:, None]
                      + np.arange(3 * (p + 1))[None, :])
    E_B, E_M1, E_M2, GE0, GE1 = shape_tensors(p)
    xi, wq, Nq, dNq = quadrature(p)

    he, xl = _geom(thick, n_per_layer, z_ref)
    Ce = C_layers[elem_layer]

    (Ke, Fe, Dee, T1e, T2e, U1e, U2e, W11e, W12e, W22e, P1e, P2e) = jax.vmap(
        _elem_blocks, in_axes=(0, 0, 0) + (None,) * 9)(
        Ce, he, xl, xi, wq, Nq, dNq, E_B, E_M1, E_M2, GE0, GE1)

    r = idx[:, :, None]; c = idx[:, None, :]
    six = jnp.arange(6)[None, None, :]

    def scat2(x):
        return jnp.zeros((ndofs, ndofs)).at[r, c].add(x)

    def scat1(x):
        return jnp.zeros((ndofs, 6)).at[idx[:, :, None], six].add(x)

    K = scat2(Ke); F = scat1(Fe)
    T1 = scat2(T1e); T2 = scat2(T2e)
    U1 = scat2(U1e); U2 = scat2(U2e)
    W11 = scat2(W11e); W12 = scat2(W12e); W22 = scat2(W22e)
    P1 = scat1(P1e); P2 = scat1(P2e)
    D_ee = jnp.sum(Dee, axis=0)

    # ---- zeroth order: warping with the rigid-translation null space projected out ----
    n_node = ndofs // 3
    null = jnp.zeros((ndofs, 3))
    null = null.at[0::3, 0].set(1.0).at[1::3, 1].set(1.0).at[2::3, 2].set(1.0)
    Q = null / jnp.sqrt(n_node)
    Pp = jnp.eye(ndofs) - Q @ Q.T
    beta = jnp.max(jnp.abs(jnp.diag(K)))
    K_proj = Pp @ K @ Pp + beta * (Q @ Q.T)

    V0 = Pp @ jnp.linalg.solve(K_proj, -(Pp @ F))
    A6 = D_ee + V0.T @ F

    # ---- first order: gradient-driven warping ----
    R1 = T1 @ V0 - (U1 @ V0 + P1)
    R2 = T2 @ V0 - (U2 @ V0 + P2)
    C1bar = Pp @ jnp.linalg.solve(K_proj, -(Pp @ R1))
    C2bar = Pp @ jnp.linalg.solve(K_proj, -(Pp @ R2))

    # ---- second order: the gradient energy ----
    H11 = V0.T @ W11 @ V0 + R1.T @ C1bar
    H12 = V0.T @ W12 @ V0 + 0.5 * (R1.T @ C2bar + C1bar.T @ R2)
    H22 = V0.T @ W22 @ V0 + R2.T @ C2bar
    H11 = 0.5 * (H11 + H11.T)
    H22 = 0.5 * (H22 + H22.T)
    H = jnp.block([[H11, H12], [H12.T, H22]])

    # ---- RM projection: E = R - D_a gamma,a, then least squares of U* ----
    D1 = jnp.zeros((6, 2)).at[3, 0].set(1.0).at[5, 1].set(1.0)
    D2 = jnp.zeros((6, 2)).at[4, 1].set(1.0).at[5, 0].set(1.0)
    S1 = null.T @ R1
    S2 = null.T @ R2
    AD1 = A6 @ D1
    AD2 = A6 @ D2

    def blocks(X, c1, c2):
        Bs = H11 + AD1 @ X @ AD1.T + c1.T @ S1 + S1.T @ c1
        Cs = H12 + AD1 @ X @ AD2.T + c1.T @ S2 + S1.T @ c2
        Ds = H22 + AD2 @ X @ AD2.T + c2.T @ S2 + S2.T @ c2
        return jnp.block([[Bs, Cs], [Cs.T, Ds]])

    b0 = -blocks(jnp.zeros((2, 2)), jnp.zeros((3, 6)), jnp.zeros((3, 6))).ravel()

    def column(j):
        pj = jnp.zeros(N_LS).at[j].set(1.0)
        X = jnp.array([[pj[0], pj[1]], [pj[1], pj[2]]])
        return blocks(X, pj[3:21].reshape(3, 6), pj[21:39].reshape(3, 6)).ravel() + b0

    Amat = jax.vmap(column)(jnp.arange(N_LS)).T
    cs = jnp.linalg.norm(Amat, axis=0)
    cs = jnp.where(cs == 0, 1.0, cs)
    sol = jnp.linalg.lstsq(Amat / cs, b0, rcond=None)[0] / cs
    X = jnp.array([[sol[0], sol[1]], [sol[1], sol[2]]])
    c1 = sol[3:21].reshape(3, 6)
    c2 = sol[21:39].reshape(3, 6)
    Ustar_rel = jnp.linalg.norm(blocks(X, c1, c2)) / (jnp.linalg.norm(H) + 1e-300)

    return dict(A6=A6, X=X, G_msg=jnp.linalg.inv(X),
                X_eigmin=jnp.min(jnp.linalg.eigvalsh(X)),
                H=H, Ustar_rel=Ustar_rel, V0=V0, C1bar=C1bar, C2bar=C2bar)


def build(thick, angles_deg, mat_names, material_db, n_per_layer=4, elem_order=3,
          z_ref=None):
    """Solve the SG for one laminate and attach its host-side mesh."""
    thick = np.asarray(thick, float)
    if z_ref is None:
        z_ref = 0.5 * thick.sum()
    C = layer_stiffness(mat_names, angles_deg, material_db)
    sol = sg_plate_solve(jnp.asarray(thick), C, int(n_per_layer), int(elem_order),
                         float(z_ref))
    m = sg_mesh(thick, int(n_per_layer), int(elem_order), float(z_ref))
    return {**sol, 'mesh': m, 'C_layers': C, 'angles': list(angles_deg),
            'mat_names': list(mat_names)}


# -------------------------------------------------------------------- recovery
def sample_points(mesh, n_per_layer_out=41, eps=1e-9):
    """Through-thickness sample points that honour ply interfaces.

    One-sided limits sit on BOTH sides of every interface, so a discontinuous stress is
    drawn as a genuine jump rather than a spurious ramp.
    """
    thick = mesh['thick']
    bot = np.concatenate([[0.0], np.cumsum(thick)]) - mesh['z_ref']
    npl_e = mesh['n_per_layer']
    zs, es = [], []
    for k in range(thick.size):
        a, b = bot[k], bot[k + 1]
        zz = np.linspace(a + eps * (b - a), b - eps * (b - a), n_per_layer_out)
        loc = np.clip(((zz - a) / (b - a) * npl_e).astype(int), 0, npl_e - 1)
        zs.append(zz)
        es.append(k * npl_e + loc)
    z = np.concatenate(zs)
    e = np.concatenate(es)
    xi = np.clip(2.0 * (z - mesh['xl'][e]) / mesh['he'][e] - 1.0, -1.0, 1.0)
    return z, e, xi


def recover(sg, E6, dE1=None, dE2=None, n_per_layer_out=41, return_warp=False):
    """3-D strain and stress through the thickness.

    ``E6`` are the plate strains and ``dE1`` / ``dE2`` their in-plane gradients -- the
    gradient terms are what supply the transverse shear.  Returns (z, Gam, Sig) in Voigt
    order [11,22,33,23,13,12]; with ``return_warp`` also the warping DISPLACEMENT
    [w1,w2,w3] at the sample points (the through-thickness fluctuation, <w> = 0), which
    is what turns the plate kinematics u0 + z*phi into the recovered 3-D displacement.
    """
    m = sg['mesh']
    p = m['p']
    E6 = jnp.asarray(E6, dtype=jnp.float64)
    dE1 = jnp.zeros(6) if dE1 is None else jnp.asarray(dE1, dtype=jnp.float64)
    dE2 = jnp.zeros(6) if dE2 is None else jnp.asarray(dE2, dtype=jnp.float64)

    z, e, xi = sample_points(m, n_per_layer_out)
    Npt, dNpt = basis_at(p, xi)
    E_B, E_M1, E_M2, GE0, GE1 = shape_tensors(p)

    idx_e = jnp.asarray(m['idx'][e])
    he_e = jnp.asarray(m['he'][e])
    lay_e = jnp.asarray(m['elem_layer'][e])
    zz = jnp.asarray(z)

    V0e = sg['V0'][idx_e]
    C1e = sg['C1bar'][idx_e]
    C2e = sg['C2bar'][idx_e]
    Ck = sg['C_layers'][lay_e]

    w_loc = V0e @ E6 + C1e @ dE1 + C2e @ dE2
    g1 = V0e @ dE1
    g2 = V0e @ dE2

    B = jnp.einsum('nj,jab->nab', dNpt * (2.0 / he_e)[:, None], E_B)
    M1 = jnp.einsum('nj,jab->nab', Npt, E_M1)
    M2 = jnp.einsum('nj,jab->nab', Npt, E_M2)
    Ge = GE0[None] + zz[:, None, None] * GE1[None]

    Gam = (jnp.einsum('nab,nb->na', B, w_loc)
           + jnp.einsum('nab,b->na', Ge, E6)
           + jnp.einsum('nab,nb->na', M1, g1)
           + jnp.einsum('nab,nb->na', M2, g2))
    Sig = jnp.einsum('nab,nb->na', Ck, Gam)
    if return_warp:
        # w_loc holds the 3*(p+1) element warping dofs; interpolate each component
        wpt = jnp.einsum('nj,nja->na', Npt,
                         w_loc.reshape(w_loc.shape[0], p + 1, 3))
        return z, Gam, Sig, wpt
    return z, Gam, Sig
