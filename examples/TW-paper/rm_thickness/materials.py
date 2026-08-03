"""materials.py -- self-contained copies of the OpenSG ply-stiffness and 1-D SG shape
function helpers, so this folder runs without importing the whole opensg_jax package.

Identical to ``opensg_jax.fe_jax.msg_materials`` (build_stiffness_6x6 / rotation_6x6 /
rotated_stiffness_6x6 / _lagrange_dN / _plate_B) -- kept byte-compatible on purpose so
any result here is directly comparable with the production pipeline.

Voigt order: [11, 22, 33, 23, 13, 12].
"""
import numpy as np


def build_stiffness_6x6(E, G, nu):
    """C = S^{-1} from orthotropic engineering constants.

    E  = [E1, E2, E3], G = [G12, G13, G23], nu = [nu12, nu13, nu23].
    """
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
    """6x6 stress-rotation for a fibre angle about the thickness axis (OpenSG R_sig)."""
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


def lagrange_N(nodes_xi, xi):
    npn = len(nodes_xi)
    N = np.ones(npn)
    for i in range(npn):
        for j in range(npn):
            if j != i:
                N[i] *= (xi - nodes_xi[j]) / (nodes_xi[i] - nodes_xi[j])
    return N


def lagrange_dN(nodes_xi, xi):
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


def plate_B(nodes_xi, xi, he):
    """gamma_h operator (6, 3*(p+1)): eps33 <- dv3/dx, gamma23 <- dv2/dx, gamma13 <- dv1/dx."""
    dN = lagrange_dN(nodes_xi, xi) * (2.0 / he)
    npn = len(nodes_xi)
    B = np.zeros((6, 3 * npn))
    for i in range(npn):
        B[2, 3 * i + 2] = dN[i]
        B[3, 3 * i + 1] = dN[i]
        B[4, 3 * i + 0] = dN[i]
    return B


def grad_ops(nodes_xi, xi):
    """M1, M2 (6, 3*(p+1)): strain from the IN-PLANE gradient of the warping field."""
    N = lagrange_N(nodes_xi, xi)
    npn = len(nodes_xi)
    M1 = np.zeros((6, 3 * npn))
    M2 = np.zeros((6, 3 * npn))
    for n in range(npn):
        M1[0, 3 * n + 0] = N[n]      # eps11 <- w1,1
        M1[4, 3 * n + 2] = N[n]      # 2g13  <- w3,1
        M1[5, 3 * n + 1] = N[n]      # g12   <- w2,1
        M2[1, 3 * n + 1] = N[n]      # eps22 <- w2,2
        M2[3, 3 * n + 2] = N[n]      # 2g23  <- w3,2
        M2[5, 3 * n + 0] = N[n]      # g12   <- w1,2
    return M1, M2


# ---------------------------------------------------------------- material sets
PAGANO = {  # Pagano's standard set, E_T taken as 1 GPa (only ratios matter)
    'E': [25.0e9, 1.0e9, 1.0e9],
    'G': [0.5e9, 0.5e9, 0.2e9],      # G12, G13, G23
    'nu': [0.25, 0.25, 0.25],
    'rho': 1.0,
}

AS4 = {  # Garg sec. 3 "4-layered" set: E1=181, E3=10.3, G13=7.17, nu13=0.28
    'E': [181.0e9, 10.3e9, 10.3e9],
    'G': [7.17e9, 7.17e9, 3.5e9],
    'nu': [0.28, 0.28, 0.28],
    'rho': 1.0,
}

SANDWICH_FACE = {  # Garg sec. 3 sandwich faces
    'E': [131.0e9, 10.34e9, 10.34e9],
    'G': [6.205e9, 6.205e9, 3.0e9],
    'nu': [0.22, 0.22, 0.22],
    'rho': 1.0,
}

SANDWICH_CORE = {  # Garg sec. 3 sandwich core
    'E': [0.5776e9, 0.5776e9, 0.5776e9],
    'G': [0.1079e9, 0.1079e9, 0.1079e9],
    'nu': [0.0025, 0.0025, 0.0025],
    'rho': 1.0,
}

MATDB = {'pagano': PAGANO, 'as4': AS4, 'face': SANDWICH_FACE, 'core': SANDWICH_CORE}
