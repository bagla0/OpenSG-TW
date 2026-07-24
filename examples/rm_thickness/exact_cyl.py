"""exact_cyl.py -- EXACT 3-D elasticity for cylindrical bending of a laminate, in JAX.

Reference solution for the whole study.  It is ANALYTICAL, not a finite element model:
each layer satisfies the 3-D equations exactly and the layer propagator is a matrix
exponential, so interface continuity and both traction boundary conditions hold to
machine precision.

This is a state-space (transfer-matrix) restatement of

    N.J. Pagano, "Exact solutions for composite laminates in cylindrical bending",
    J. Compos. Mater. 3 (1969) 398-411,

and generalises it in one respect: Pagano builds a stress function f_i(z) from the roots
of a quartic that exists only for a specially orthotropic (cross-ply) layer, whereas the
state-space form carries a six-component state and therefore admits ANY monoclinic ply,
i.e. arbitrary fibre angles.

PROBLEM.  Thickness h, infinite in y (d/dy = 0), simply supported at x = 0, L, loaded on
the top face by sigma33(x, +h/2) = q0 sin(p x), p = n*pi/L, everything else traction-free.
Layer 0 is at the BOTTOM.

STATE.  With
    u = U(z) cos(px),  v = V(z) cos(px),  w = W(z) sin(px),
    sigma13 = X(z) cos(px),  sigma23 = Y(z) cos(px),  sigma33 = Z(z) sin(px),
the 3-D field equations reduce EXACTLY to  s' = A s,  s = [U, V, W, X, Y, Z],  with A
constant inside a layer:

    U' = (Gs^-1 [X;Y])_1 - p W                Gs = [[C55, C45], [C45, C44]]
    V' = (Gs^-1 [X;Y])_2
    W' = (p C13 U + p C36 V + Z) / C33
    X' = p^2 Q11 U + p^2 Q16 V - p (C13/C33) Z      Q11 = C11 - C13^2/C33
    Y' = p^2 Q16 U + p^2 Q66 V - p (C36/C33) Z      Q16 = C16 - C13 C36/C33
    Z' = p X                                        Q66 = C66 - C36^2/C33

SCALING (essential).  In physical units A mixes entries of order 1/C ~ 1e-11 with entries
of order p^2 C ~ 1e10, a spread of ~1e21.  ``expm`` of such a matrix is meaningless even
though its eigenvalues are O(p): the scaling-and-squaring recursion squares a strongly
non-normal matrix ~30 times.  So the system is solved in the dimensionless variables

    z~ = z/h,   (U,V,W)~ = (U,V,W)/h,   (X,Y,Z)~ = (X,Y,Z)/E0,   p~ = p h,

with E0 a reference modulus, which makes every entry of A~ order unity.  Physical stress
is E0 * (tilde) and physical displacement h * (tilde).

Voigt order is OpenSG's [11, 22, 33, 23, 13, 12].
"""
import numpy as np
from jax.scipy.linalg import expm

from jaxcfg import jax, jnp
from materials import layer_stiffness


def _layer_A(C, pt, E0):
    """Dimensionless state matrix A~ (6x6) for one monoclinic layer."""
    C11, C13, C16 = C[0, 0], C[0, 2], C[0, 5]
    C33, C36 = C[2, 2], C[2, 5]
    C44, C45, C55 = C[3, 3], C[3, 4], C[4, 4]
    C66 = C[5, 5]
    Gi = jnp.linalg.inv(jnp.array([[C55, C45], [C45, C44]])) * E0
    Q11 = (C11 - C13 * C13 / C33) / E0
    Q16 = (C16 - C13 * C36 / C33) / E0
    Q66 = (C66 - C36 * C36 / C33) / E0
    A = jnp.zeros((6, 6))
    A = A.at[0, 2].set(-pt).at[0, 3].set(Gi[0, 0]).at[0, 4].set(Gi[0, 1])
    A = A.at[1, 3].set(Gi[1, 0]).at[1, 4].set(Gi[1, 1])
    A = A.at[2, 0].set(pt * C13 / C33).at[2, 1].set(pt * C36 / C33)
    A = A.at[2, 5].set(E0 / C33)
    A = A.at[3, 0].set(pt * pt * Q11).at[3, 1].set(pt * pt * Q16)
    A = A.at[3, 5].set(-pt * C13 / C33)
    A = A.at[4, 0].set(pt * pt * Q16).at[4, 1].set(pt * pt * Q66)
    A = A.at[4, 5].set(-pt * C36 / C33)
    A = A.at[5, 3].set(pt)
    return A


@jax.jit
def _solve_state(t_hat, C_layers, pt, E0, q_hat):
    """Bottom-face state and cumulative transfer matrices, all dimensionless."""
    A = jax.vmap(_layer_A, in_axes=(0, None, None))(C_layers, pt, E0)
    Tk = jax.vmap(lambda a, t: expm(a * t))(A, t_hat)

    def step(T, Tk_i):
        return Tk_i @ T, T                       # carry, cumulative-to-layer-bottom

    T_tot, T_cum = jax.lax.scan(step, jnp.eye(6), Tk)
    uvw0 = jnp.linalg.solve(T_tot[3:6, 0:3], jnp.array([0.0, 0.0, q_hat]))
    s0 = jnp.concatenate([uvw0, jnp.zeros(3)])
    return s0, A, T_cum, T_tot


@jax.jit
def _fields(z_hat, lay, s0, A, T_cum, zb_hat, C_layers, pt, E0, h):
    """Physical stress/strain/displacement amplitudes at sample points."""
    def one(zi, ki):
        s = expm(A[ki] * (zi - zb_hat[ki])) @ (T_cum[ki] @ s0)
        U, V, W, X, Y, Z = s                                   # dimensionless
        C = C_layers[ki]
        C11, C12, C13, C16 = C[0, 0], C[0, 1], C[0, 2], C[0, 5]
        C23, C26 = C[1, 2], C[1, 5]
        C33, C36, C66 = C[2, 2], C[2, 5], C[5, 5]
        e33 = (pt * C13 * U + pt * C36 * V + E0 * Z) / C33
        S11 = -pt * C11 * U - pt * C16 * V + C13 * e33
        S22 = -pt * C12 * U - pt * C26 * V + C23 * e33
        S12 = -pt * C16 * U - pt * C66 * V + C36 * e33
        Gs = jnp.array([[C[4, 4], C[3, 4]], [C[3, 4], C[3, 3]]])
        g13, g23 = jnp.linalg.solve(Gs, E0 * jnp.array([X, Y]))
        sig = jnp.array([S11, S22, E0 * Z, E0 * Y, E0 * X, S12])
        eps = jnp.array([-pt * U, 0.0, e33, g23, g13, -pt * V])
        return sig, eps, h * jnp.array([U, V, W])

    return jax.vmap(one)(z_hat, lay)


class ExactCyl:
    """Exact cylindrical-bending solution.  Layer 0 = bottom face.

    Amplitudes: sigma_{11,22,33,12} multiply sin(px); sigma_{13,23} multiply cos(px).
    """

    def __init__(self, thick, angles_deg, mat_names, material_db, L, q0=1.0, n=1):
        self.thick = np.asarray(thick, float)
        self.h = float(self.thick.sum())
        self.L = float(L)
        self.q0 = float(q0)
        self.p = float(n * np.pi / self.L)
        self.C = layer_stiffness(mat_names, angles_deg, material_db)
        self.z_bot = np.concatenate([[0.0], np.cumsum(self.thick)])

        self.E0 = float(jnp.max(jnp.abs(self.C)))
        self.pt = self.p * self.h
        self.q_hat = self.q0 / self.E0
        self.s0, self.A, self.T_cum, self.T_tot = _solve_state(
            jnp.asarray(self.thick / self.h), self.C, self.pt, self.E0, self.q_hat)

    def _lay(self, z):
        return np.clip(np.searchsorted(self.z_bot[1:-1], z, side='left'), 0,
                       self.thick.size - 1).astype(int)

    def at(self, z, lay=None):
        """(sig, eps, uvw) at through-thickness coordinates ``z`` measured from BOTTOM."""
        z = np.atleast_1d(np.asarray(z, float))
        lay = self._lay(z) if lay is None else np.asarray(lay, int)
        sig, eps, uvw = _fields(jnp.asarray(z / self.h), jnp.asarray(lay), self.s0,
                                self.A, self.T_cum,
                                jnp.asarray(self.z_bot[:-1] / self.h),
                                self.C, self.pt, self.E0, self.h)
        return np.asarray(sig), np.asarray(eps), np.asarray(uvw)

    def profile(self, n_per_layer=41, eps=1e-12):
        """Interface-honouring through-thickness sample, z referred to MID-surface."""
        zs, lay = [], []
        for k in range(self.thick.size):
            a, b = self.z_bot[k], self.z_bot[k + 1]
            zs.append(np.linspace(a + eps * (b - a), b - eps * (b - a), n_per_layer))
            lay.append(np.full(n_per_layer, k))
        zs = np.concatenate(zs); lay = np.concatenate(lay)
        sig, eps6, uvw = self.at(zs, lay)
        return zs - self.h / 2.0, sig, eps6, uvw

    # ------------------------------------------------------------- verification
    def bc_residual(self):
        """max traction-BC violation / q0 (machine precision by construction)."""
        sb = self.T_cum[0] @ self.s0
        st = self.T_tot @ self.s0
        r = jnp.array([sb[3], sb[4], sb[5], st[3], st[4], st[5] - self.q_hat])
        return float(jnp.max(jnp.abs(r)) / abs(self.q_hat))

    def resultants(self, n_gauss=200):
        """[N11, M11, Q1] from the exact stress.  Must equal [0, q0/p^2, q0/p]."""
        xi, wq = np.polynomial.legendre.leggauss(n_gauss)
        N11 = M11 = Q1 = 0.0
        for k in range(self.thick.size):
            a, b = self.z_bot[k], self.z_bot[k + 1]
            zq = 0.5 * (a + b) + 0.5 * (b - a) * xi
            wt = 0.5 * (b - a) * wq
            sig, _, _ = self.at(zq, np.full(zq.size, k))
            N11 += float(np.sum(wt * sig[:, 0]))
            M11 += float(np.sum(wt * sig[:, 0] * (zq - self.h / 2.0)))
            Q1 += float(np.sum(wt * sig[:, 4]))
        return N11, M11, Q1
