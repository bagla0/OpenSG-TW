"""materials.py -- ply stiffness and 1-D SG shape-function tensors, in JAX.

The ply stiffness follows ``opensg_jax.fe_jax.msg_materials`` exactly (same Voigt order
[11,22,33,23,13,12], same R_sig rotation convention), so any number produced here is
directly comparable with the production OpenSG pipeline.

The shape-function part is arranged as CONSTANT tensors so that the element operators can
be formed by a single contraction inside a jitted, vmapped kernel:

    B  = einsum('i,iab->ab', dN * 2/he, E_B)      through-thickness  gamma_h
    M1 = einsum('i,iab->ab', N,         E_M1)     in-plane x1 gradient of the warping
    M2 = einsum('i,iab->ab', N,         E_M2)     in-plane x2 gradient of the warping
    Ge = GE0 + x * GE1                            plate-strain -> 3-D strain
"""
import numpy as np

from jaxcfg import jnp

VOIGT = ("11", "22", "33", "23", "13", "12")


# --------------------------------------------------------------------- stiffness
def build_stiffness_6x6(E, G, nu):
    """C = S^{-1};  E = [E1,E2,E3], G = [G12,G13,G23], nu = [nu12,nu13,nu23]."""
    E1, E2, E3 = E
    G12, G13, G23 = G
    v12, v13, v23 = nu
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E3
    S[0, 1] = S[1, 0] = -v12 / E1
    S[0, 2] = S[2, 0] = -v13 / E1
    S[1, 2] = S[2, 1] = -v23 / E2
    S[3, 3] = 1.0 / G23
    S[4, 4] = 1.0 / G13
    S[5, 5] = 1.0 / G12
    return np.linalg.inv(S)


def rotation_6x6(theta_deg):
    """OpenSG R_sig for a fibre angle about the thickness axis."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    cs = c * s
    return np.array([
        [c ** 2,  s ** 2, 0, 0, 0, -2 * cs],
        [s ** 2,  c ** 2, 0, 0, 0,  2 * cs],
        [0,       0,      1, 0, 0,  0],
        [0,       0,      0, c, s,  0],
        [0,       0,      0, -s, c, 0],
        [cs,     -cs,     0, 0, 0,  c ** 2 - s ** 2],
    ])


def rotated_stiffness_6x6(E, G, nu, theta_deg):
    C = build_stiffness_6x6(E, G, nu)
    R = rotation_6x6(theta_deg)
    return R @ C @ R.T


def layer_stiffness(mat_names, angles_deg, material_db):
    """(nlay, 6, 6) jnp array of rotated ply stiffnesses."""
    return jnp.asarray(np.stack([
        rotated_stiffness_6x6(material_db[m]['E'], material_db[m]['G'],
                              material_db[m]['nu'], a)
        for m, a in zip(mat_names, angles_deg)]))


# ------------------------------------------------------------- shape functions
def _lagrange_N(nodes_xi, xi):
    npn = len(nodes_xi)
    N = np.ones(npn)
    for i in range(npn):
        for j in range(npn):
            if j != i:
                N[i] *= (xi - nodes_xi[j]) / (nodes_xi[i] - nodes_xi[j])
    return N


def _lagrange_dN(nodes_xi, xi):
    n = len(nodes_xi)
    dN = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j == i:
                continue
            term = 1.0 / (nodes_xi[i] - nodes_xi[j])
            for m in range(n):
                if m == i or m == j:
                    continue
                term *= (xi - nodes_xi[m]) / (nodes_xi[i] - nodes_xi[m])
            s += term
        dN[i] = s
    return dN


def shape_tensors(p):
    """Constant selection tensors for a p-th order 1-D element.

    E_B[i]  : eps33 <- dv3/dx, gamma23 <- dv2/dx, gamma13 <- dv1/dx   (row 2,3,4)
    E_M1[i] : eps11 <- w1,1  ; 2g13 <- w3,1 ; g12 <- w2,1             (row 0,4,5)
    E_M2[i] : eps22 <- w2,2  ; 2g23 <- w3,2 ; g12 <- w1,2             (row 1,3,5)
    """
    nd = 3 * (p + 1)
    E_B = np.zeros((p + 1, 6, nd))
    E_M1 = np.zeros((p + 1, 6, nd))
    E_M2 = np.zeros((p + 1, 6, nd))
    for i in range(p + 1):
        E_B[i, 2, 3 * i + 2] = 1.0
        E_B[i, 3, 3 * i + 1] = 1.0
        E_B[i, 4, 3 * i + 0] = 1.0
        E_M1[i, 0, 3 * i + 0] = 1.0
        E_M1[i, 4, 3 * i + 2] = 1.0
        E_M1[i, 5, 3 * i + 1] = 1.0
        E_M2[i, 1, 3 * i + 1] = 1.0
        E_M2[i, 3, 3 * i + 2] = 1.0
        E_M2[i, 5, 3 * i + 0] = 1.0
    GE0 = np.zeros((6, 6)); GE1 = np.zeros((6, 6))
    GE0[0, 0] = GE0[1, 1] = GE0[5, 2] = 1.0
    GE1[0, 3] = GE1[1, 4] = GE1[5, 5] = 1.0
    return (jnp.asarray(E_B), jnp.asarray(E_M1), jnp.asarray(E_M2),
            jnp.asarray(GE0), jnp.asarray(GE1))


def quadrature(p, n_extra=1):
    """Gauss points + the basis sampled there.  Returns (xi, w, N, dN) as jnp arrays."""
    ng = max(3, p + n_extra)
    xi, w = np.polynomial.legendre.leggauss(ng)
    nodes = np.linspace(-1.0, 1.0, p + 1)
    N = np.stack([_lagrange_N(nodes, x) for x in xi])
    dN = np.stack([_lagrange_dN(nodes, x) for x in xi])
    return jnp.asarray(xi), jnp.asarray(w), jnp.asarray(N), jnp.asarray(dN)


def basis_at(p, xi):
    """Basis and its derivative at arbitrary xi (numpy scalars/arrays -> jnp)."""
    nodes = np.linspace(-1.0, 1.0, p + 1)
    xi = np.atleast_1d(np.asarray(xi, float))
    N = np.stack([_lagrange_N(nodes, x) for x in xi])
    dN = np.stack([_lagrange_dN(nodes, x) for x in xi])
    return jnp.asarray(N), jnp.asarray(dN)


# ---------------------------------------------------------------- material sets
PAGANO = {   # Pagano's standard set: E_L/E_T = 25, G_LT/E_T = 0.5, G_TT/E_T = 0.2, nu = 0.25
    'E': [25.0e9, 1.0e9, 1.0e9],
    'G': [0.5e9, 0.5e9, 0.2e9],
    'nu': [0.25, 0.25, 0.25],
    'rho': 1.0,
}

AS4 = {      # Garg sec. 3, four-layer cases:  E1 = 181, E3 = 10.3, G13 = 7.17, nu13 = 0.28
    'E': [181.0e9, 10.3e9, 10.3e9],
    'G': [7.17e9, 7.17e9, 3.5e9],
    'nu': [0.28, 0.28, 0.28],
    'rho': 1.0,
}

SANDWICH_FACE = {   # Garg sec. 3 sandwich faces
    'E': [131.0e9, 10.34e9, 10.34e9],
    'G': [6.205e9, 6.205e9, 3.0e9],
    'nu': [0.22, 0.22, 0.22],
    'rho': 1.0,
}

SANDWICH_CORE = {   # Garg sec. 3 sandwich core
    'E': [0.5776e9, 0.5776e9, 0.5776e9],
    'G': [0.1079e9, 0.1079e9, 0.1079e9],
    'nu': [0.0025, 0.0025, 0.0025],
    'rho': 1.0,
}

ISO = {'E': [70.0e9] * 3, 'G': [70.0e9 / 2.6] * 3, 'nu': [0.3] * 3, 'rho': 1.0}
ISO0 = {'E': [70.0e9] * 3, 'G': [35.0e9] * 3, 'nu': [0.0] * 3, 'rho': 1.0}

MATDB = {'pagano': PAGANO, 'as4': AS4, 'face': SANDWICH_FACE, 'core': SANDWICH_CORE,
         'iso': ISO, 'iso0': ISO0}
