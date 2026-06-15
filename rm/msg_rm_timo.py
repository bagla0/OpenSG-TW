"""
RM Timoshenko (V1) step: produce the 6x6 stiffness (incl. C22, C33) from the
RM C0 cross-section model, reusing the validated msg_solver condensation.

Pipeline (mirrors the Kirchhoff path):
  EB:   Dhh,Dhe,Dee  + constraints C, rigid kernel Psi  -> V0, D_eff(4x4)
  V1:   eps_l (shear-warping) -> Dhl,Dll,Dle -> V1 -> 6x6 [EA,GA12,GA13,GJ,EI2,EI3]

RM rigid kernel (Psi, derived from eq 4.23, zero-strain modes):
  3 translations (w1,w2,w3=const) + twist (w2=-y3, w3=y2, omega1=-1).
Constraints C: <w1>=<w2>=<w3>=<omega1>=0 (conjugate to Psi).
"""
import os, sys
import numpy as np
from scipy.sparse import coo_matrix
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from fe_jax.msg_solver import (solve_fluctuation_field, prepare_v1_rhs,
                               finalize_v1_and_compute_deff)
from msg_rm import _lagrange, _shape, _macro_BD, _macro_BG


def _elem_BD_BG_BL(nodes_xi, xi, X, dN_unused, k22, p):
    """Return BDq(6,5n), BGq(2,5n), BLq(6,5n) and geometry at xi."""
    N, dNr = _shape(nodes_xi, xi)
    dxds = dNr @ X; Jac = np.hypot(*dxds); dN = dNr / Jac
    x2, x3 = N @ X; t2, t3 = dxds / Jac; n2, n3 = t3, -t2
    npn = p + 1
    BDq = np.zeros((6, 5*npn)); BGq = np.zeros((2, 5*npn)); BLq = np.zeros((6, 5*npn))
    for a in range(npn):
        o = 5*a
        # EB warping strain (eps_h)
        BDq[1, o+1] += t2*dN[a]; BDq[1, o+2] += t3*dN[a]
        BDq[2, o+0] += dN[a]
        BDq[4, o+3] += -dN[a]
        BDq[5, o+4] += dN[a]; BDq[5, o+0] += 0.5*k22*dN[a]
        BGq[0, o+4] += N[a]
        BGq[1, o+1] += n2*dN[a]; BGq[1, o+2] += n3*dN[a]; BGq[1, o+3] += -N[a]
        # shear-warping operator (eps_l), Kirchhoff-style on the w DOFs
        BLq[0, o+0] += N[a]                                   # eps11 = w1
        BLq[2, o+1] += t2*N[a]; BLq[2, o+2] += t3*N[a]        # 2eps12 = t.w
        BLq[5, o+1] += 2*t3*dN[a] - 0.5*k22*t2*N[a]           # kappa12
        BLq[5, o+2] += -2*t2*dN[a] - 0.5*k22*t3*N[a]
    return BDq, BGq, BLq, (x2, x3, t2, t3, Jac)


def assemble_all(nodes, elems, layup_per_elem, D_by, G_by, k22_e, p=1, reduced=True):
    Nn = len(nodes); ndof = 5*Nn
    nodes_xi = _lagrange(p)
    Dhh = np.zeros((ndof, ndof)); Dhe = np.zeros((ndof, 4)); Dee = np.zeros((4, 4))
    Dhl = np.zeros((ndof, ndof)); Dll = np.zeros((ndof, ndof)); Dle = np.zeros((ndof, 4))
    xgD, wgD = np.polynomial.legendre.leggauss(p+1)
    xgG, wgG = np.polynomial.legendre.leggauss(max(1, p))
    for e, el in enumerate(elems):
        X = nodes[el]; k22 = float(k22_e[e]); ln = layup_per_elem[e]
        D = D_by[ln]; G = G_by[ln]
        g = np.concatenate([[5*nd, 5*nd+1, 5*nd+2, 5*nd+3, 5*nd+4] for nd in el])
        # D-energy (full int) + eps_l matrices
        for xi, w in zip(xgD, wgD):
            BDq, BGq, BLq, geo = _elem_BD_BG_BL(nodes_xi, xi, X, None, k22, p)
            x2, x3, t2, t3, Jac = geo; dl = Jac*w
            BDe = _macro_BD(x2, x3, t2, t3, k22)
            Dhh[np.ix_(g, g)] += BDq.T @ D @ BDq * dl
            Dhe[g] += BDq.T @ D @ BDe * dl
            Dee += BDe.T @ D @ BDe * dl
            Dhl[np.ix_(g, g)] += BDq.T @ D @ BLq * dl
            Dll[np.ix_(g, g)] += BLq.T @ D @ BLq * dl
            Dle[g] += BLq.T @ D @ BDe * dl
        # G-energy (reduced int)
        for xi, w in zip((xgG, xgD)[not reduced], (wgG, wgD)[not reduced]):
            BDq, BGq, BLq, geo = _elem_BD_BG_BL(nodes_xi, xi, X, None, k22, p)
            x2, x3, t2, t3, Jac = geo; dl = Jac*w
            BGe = _macro_BG(x2, x3, t2, t3)
            Dhh[np.ix_(g, g)] += BGq.T @ G @ BGq * dl
            Dhe[g] += BGq.T @ G @ BGe * dl
            Dee += BGe.T @ G @ BGe * dl
    return Dhh, Dhe, Dee, Dhl, Dll, Dle


def build_C_Psi(nodes, elems, p=1):
    Nn = len(nodes); ndof = 5*Nn
    nodes_xi = _lagrange(p)
    C = np.zeros((4, ndof)); Psi = np.zeros((ndof, 4))
    xg, wg = np.polynomial.legendre.leggauss(p+1)
    for el in elems:
        X = nodes[el]
        for xi, w in zip(xg, wg):
            N, dNr = _shape(nodes_xi, xi); Jac = np.hypot(*(dNr @ X)); dl = Jac*w
            for a, nd in enumerate(el):
                C[0, 5*nd+0] += N[a]*dl          # <w1>
                C[1, 5*nd+1] += N[a]*dl          # <w2>
                C[2, 5*nd+2] += N[a]*dl          # <w3>
                C[3, 5*nd+3] += N[a]*dl          # <omega1>
    for nd in range(Nn):
        y2, y3 = nodes[nd]
        Psi[5*nd+0, 0] = 1.0                     # w1 translation
        Psi[5*nd+1, 1] = 1.0                     # w2 translation
        Psi[5*nd+2, 2] = 1.0                     # w3 translation
        Psi[5*nd+1, 3] = -y3; Psi[5*nd+2, 3] = y2; Psi[5*nd+3, 3] = -1.0  # twist
    return C, Psi


def timoshenko_rm(nodes, elems, layup_per_elem, D_by, G_by, k22_e, p=1, reduced=True):
    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_all(
        nodes, elems, layup_per_elem, D_by, G_by, k22_e, p, reduced)
    C, Psi = build_C_Psi(nodes, elems, p)
    Dc = C.T
    Dhh_coo = coo_matrix(Dhh)
    V0, D1, A_aug = solve_fluctuation_field(Dhh_coo, -Dhe, Dc)
    Deff = Dee + np.asarray(D1)
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle),
        jnp.array(Psi), jnp.array(Dc))
    n = Dhh.shape[0]
    R_v1 = np.concatenate([np.array(bb), np.zeros((4, bb.shape[1]))], axis=0)
    import pypardiso
    V_aug = pypardiso.spsolve(A_aug, R_v1)
    C6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V_aug[:n, :]), jnp.array(V0), jnp.array(Deff),
        V0DllV0, DhlV0, DhlTV0Dle, jnp.array(Psi), jnp.array(Dc))
    return np.asarray(C6), np.asarray(Deff)
