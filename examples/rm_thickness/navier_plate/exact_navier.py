"""exact_navier.py -- EXACT 3-D elasticity for the simply supported RECTANGULAR cross-ply
plate under a doubly-sinusoidal transverse load, in JAX.

This is the benchmark configuration of Mendonca & Ruviaro, Finite Elem. Anal. Des. 260
(2026) 104609, whose reference is

    N.J. Pagano, "Exact solutions for rectangular bidirectional composites and sandwich
    plates", J. Compos. Mater. 4 (1970) 20-34.

State-space (transfer-matrix) restatement, exactly as ``../exact_cyl.py`` did for
cylindrical bending, now with TWO wavenumbers.

PROBLEM.  Plate 0<=x<=a, 0<=y<=b, 0<=z<=H (layer 1 at the BOTTOM), simply supported on
all four edges, loaded on the top face by

    sigma_z(x, y, H) = q11 sin(px) sin(qy),      p = pi/a,  q = pi/b,

with the bottom face and all face shear tractions free.  Every layer is orthotropic in
the GLOBAL axes (0 or 90 deg plies, or a transversely isotropic core), i.e.
C16 = C26 = C36 = C45 = 0.

STATE.  With the Navier form
    u = U(z) cos(px) sin(qy),   v = V(z) sin(px) cos(qy),   w = W(z) sin(px) sin(qy),
    tau_xz = X(z) cos(px) sin(qy),  tau_yz = Y(z) sin(px) cos(qy),
    sigma_z = Z(z) sin(px) sin(qy),
the 3-D equations reduce EXACTLY to s' = A s with s = [U,V,W,X,Y,Z] and, per layer,

    U' = X/C55 - p W
    V' = Y/C44 - q W
    W' = (p C13 U + q C23 V + Z)/C33
    X' = (p^2 Qb11 + q^2 C66) U + pq (Qb12 + C66) V - (p C13/C33) Z
    Y' = pq (Qb12 + C66) U + (p^2 C66 + q^2 Qb22) V - (q C23/C33) Z
    Z' = p X + q Y

with Qb11 = C11 - C13^2/C33, Qb12 = C12 - C13 C23/C33, Qb22 = C22 - C23^2/C33.

Solved in the dimensionless variables z~=z/H, (U,V,W)~=(U,V,W)/H, (X,Y,Z)~=(X,Y,Z)/E0,
p~=pH, q~=qH -- the same scaling that keeps ``expm`` meaningful (see ../exact_cyl.py).

Voigt order [11,22,33,23,13,12].  Amplitude families:
  sigma_{x,y,z}, tau_xy->CC?  no:  sigma_x, sigma_y, sigma_z, eps: SS;  tau_xy: CC;
  tau_xz, u: CS;   tau_yz, v: SC;   w: SS.
"""
import sys
import os

import numpy as np
from jax.scipy.linalg import expm

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, '..') not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, '..'))

from jaxcfg import jax, jnp                          # noqa: E402
from materials import layer_stiffness                # noqa: E402


def _layer_A(C, pt, qt, E0):
    """Dimensionless state matrix for one globally-orthotropic layer."""
    C11, C12, C13 = C[0, 0], C[0, 1], C[0, 2]
    C22, C23, C33 = C[1, 1], C[1, 2], C[2, 2]
    C44, C55, C66 = C[3, 3], C[4, 4], C[5, 5]
    Qb11 = (C11 - C13 * C13 / C33) / E0
    Qb12 = (C12 - C13 * C23 / C33) / E0
    Qb22 = (C22 - C23 * C23 / C33) / E0
    C66n = C66 / E0
    A = jnp.zeros((6, 6))
    A = A.at[0, 3].set(E0 / C55).at[0, 2].set(-pt)
    A = A.at[1, 4].set(E0 / C44).at[1, 2].set(-qt)
    A = A.at[2, 0].set(pt * C13 / C33).at[2, 1].set(qt * C23 / C33)
    A = A.at[2, 5].set(E0 / C33)
    A = A.at[3, 0].set(pt * pt * Qb11 + qt * qt * C66n)
    A = A.at[3, 1].set(pt * qt * (Qb12 + C66n))
    A = A.at[3, 5].set(-pt * C13 / C33)
    A = A.at[4, 0].set(pt * qt * (Qb12 + C66n))
    A = A.at[4, 1].set(pt * pt * C66n + qt * qt * Qb22)
    A = A.at[4, 5].set(-qt * C23 / C33)
    A = A.at[5, 3].set(pt).at[5, 4].set(qt)
    return A


@jax.jit
def _solve_state(t_hat, C_layers, pt, qt, E0, q_hat):
    A = jax.vmap(_layer_A, in_axes=(0, None, None, None))(C_layers, pt, qt, E0)
    Tk = jax.vmap(lambda a, t: expm(a * t))(A, t_hat)
    T_tot, T_cum = jax.lax.scan(lambda T, Ti: (Ti @ T, T), jnp.eye(6), Tk)
    uvw0 = jnp.linalg.solve(T_tot[3:6, 0:3], jnp.array([0.0, 0.0, q_hat]))
    s0 = jnp.concatenate([uvw0, jnp.zeros(3)])
    return s0, A, T_cum, T_tot


@jax.jit
def _fields(z_hat, lay, s0, A, T_cum, zb_hat, C_layers, pt, qt, E0, H):
    """Physical amplitudes at sample points: (sig6, eps6, uvw)."""
    def one(zi, ki):
        s = expm(A[ki] * (zi - zb_hat[ki])) @ (T_cum[ki] @ s0)
        U, V, W, X, Y, Z = s
        C = C_layers[ki]
        C11, C12, C13 = C[0, 0], C[0, 1], C[0, 2]
        C22, C23, C33 = C[1, 1], C[1, 2], C[2, 2]
        C66 = C[5, 5]
        # strain amplitudes (dimensionless state -> physical strains are O(1) already:
        # eps_x = u,x = -p U cos->  amplitude -pt*U (since U~ = U/H, p = pt/H)
        ex = -pt * U
        ey = -qt * V
        gxy = qt * U + pt * V                      # CC family
        ez = (pt * C13 * U + qt * C23 * V + E0 * Z) / C33
        Sx = C11 * ex + C12 * ey + C13 * ez
        Sy = C12 * ex + C22 * ey + C23 * ez
        Sxy = C66 * gxy
        Gs = jnp.array([[C[4, 4], C[3, 4]], [C[3, 4], C[3, 3]]])
        g13, g23 = jnp.linalg.solve(Gs, E0 * jnp.array([X, Y]))
        sig = jnp.array([Sx, Sy, E0 * Z, E0 * Y, E0 * X, Sxy])
        eps = jnp.array([ex, ey, ez, g23, g13, gxy])
        return sig, eps, H * jnp.array([U, V, W])

    return jax.vmap(one)(z_hat, lay)


class ExactNavier:
    """Exact 3-D solution for the double-sine loaded cross-ply plate.

    Amplitude families: u, tau_xz -> cos(px)sin(qy);  v, tau_yz -> sin(px)cos(qy);
    w, sigma_{x,y,z}, eps -> sin(px)sin(qy);  tau_xy -> cos(px)cos(qy).
    """

    def __init__(self, thick, angles_deg, mat_names, material_db, a, b, q11=1.0,
                 m=1, n=1):
        self.thick = np.asarray(thick, float)
        self.H = float(self.thick.sum())
        self.a, self.b = float(a), float(b)
        self.q11 = float(q11)
        self.p = m * np.pi / self.a
        self.q = n * np.pi / self.b
        self.C = layer_stiffness(mat_names, angles_deg, material_db)
        self.z_bot = np.concatenate([[0.0], np.cumsum(self.thick)])

        self.E0 = float(jnp.max(jnp.abs(self.C)))
        self.pt = self.p * self.H
        self.qt = self.q * self.H
        self.q_hat = self.q11 / self.E0
        self.s0, self.A, self.T_cum, self.T_tot = _solve_state(
            jnp.asarray(self.thick / self.H), self.C, self.pt, self.qt, self.E0,
            self.q_hat)

    def _lay(self, z):
        return np.clip(np.searchsorted(self.z_bot[1:-1], z, side='left'), 0,
                       self.thick.size - 1).astype(int)

    def at(self, z, lay=None):
        z = np.atleast_1d(np.asarray(z, float))
        lay = self._lay(z) if lay is None else np.asarray(lay, int)
        sig, eps, uvw = _fields(jnp.asarray(z / self.H), jnp.asarray(lay), self.s0,
                                self.A, self.T_cum,
                                jnp.asarray(self.z_bot[:-1] / self.H),
                                self.C, self.pt, self.qt, self.E0, self.H)
        return np.asarray(sig), np.asarray(eps), np.asarray(uvw)

    def profile(self, n_per_layer=41, eps=1e-12):
        """(zc, sig, eps6, uvw) with zc from the MID-surface, interfaces doubled."""
        zs, lay = [], []
        for k in range(self.thick.size):
            za, zb = self.z_bot[k], self.z_bot[k + 1]
            zs.append(np.linspace(za + eps * (zb - za), zb - eps * (zb - za),
                                  n_per_layer))
            lay.append(np.full(n_per_layer, k))
        zs = np.concatenate(zs); lay = np.concatenate(lay)
        sig, eps6, uvw = self.at(zs, lay)
        return zs - self.H / 2.0, sig, eps6, uvw

    # ------------------------------------------------------------- verification
    def bc_residual(self):
        sb = self.T_cum[0] @ self.s0
        st = self.T_tot @ self.s0
        r = jnp.array([sb[3], sb[4], sb[5], st[3], st[4], st[5] - self.q_hat])
        return float(jnp.max(jnp.abs(r)) / abs(self.q_hat))

    def sigz_closure(self, n_gauss=300):
        """Rebuild sigma_z by integrating p*X + q*Y; must land on q11 at the top."""
        xi, wq = np.polynomial.legendre.leggauss(n_gauss)
        acc = 0.0
        for k in range(self.thick.size):
            za, zb = self.z_bot[k], self.z_bot[k + 1]
            zq = 0.5 * (za + zb) + 0.5 * (zb - za) * xi
            wt = 0.5 * (zb - za) * wq
            sig, _, _ = self.at(zq, np.full(zq.size, k))
            acc += float(np.sum(wt * (self.p * sig[:, 4] + self.q * sig[:, 3])))
        return abs(acc / self.q11 - 1.0)
