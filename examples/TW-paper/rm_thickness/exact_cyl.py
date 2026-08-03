"""exact_cyl.py -- EXACT 3-D elasticity for cylindrical bending of a laminate.

Generalisation of Pagano (1969, J. Compos. Mater. 3:398) obtained by a state-space
(transfer-matrix) formulation instead of the quartic-root construction.  Advantages:

  * works for ANY monoclinic ply (arbitrary fibre angle), not only cross-ply -- Pagano's
    original f(z) construction is restricted to specially-orthotropic layers;
  * every layer propagator is a matrix exponential, so interface continuity of
    (u, v, w, sigma33, sigma13, sigma23) and the traction BCs are satisfied to machine
    precision by construction -- no root-finding, no ill-conditioned 4m x 4m system.

PROBLEM.  Laminate of total thickness h, infinite in y (d/dy = 0), simply supported at
x = 0, L, loaded on the TOP face by  sigma33(x, h) = q0 sin(p x),  p = n*pi/L, with the
bottom face and both shear tractions traction-free.  Layer 0 is at the BOTTOM.

STATE VECTOR.  With
    u = U(z) cos(px),  v = V(z) cos(px),  w = W(z) sin(px),
    sigma13 = X(z) cos(px),  sigma23 = Y(z) cos(px),  sigma33 = Z(z) sin(px),
the 3-D equations reduce EXACTLY to  s' = A s  with  s = [U, V, W, X, Y, Z].

Voigt order is OpenSG's [11, 22, 33, 23, 13, 12].
"""
import numpy as np
from scipy.linalg import expm

from materials import rotated_stiffness_6x6


def _layer_A(C, p):
    """State matrix A (6x6) for one monoclinic layer, s = [U, V, W, X, Y, Z]."""
    C11, C12, C13, C16 = C[0, 0], C[0, 1], C[0, 2], C[0, 5]
    C22, C23, C26 = C[1, 1], C[1, 2], C[1, 5]
    C33, C36 = C[2, 2], C[2, 5]
    C44, C45, C55 = C[3, 3], C[3, 4], C[4, 4]
    C66 = C[5, 5]

    # [sigma13; sigma23] = Gs [gamma13; gamma23]
    Gs = np.array([[C55, C45], [C45, C44]])
    Gi = np.linalg.inv(Gs)

    # sigma33-condensed in-plane moduli
    Q11 = C11 - C13 * C13 / C33
    Q16 = C16 - C13 * C36 / C33
    Q66 = C66 - C36 * C36 / C33

    A = np.zeros((6, 6))
    A[0, 2] = -p;            A[0, 3] = Gi[0, 0]; A[0, 4] = Gi[0, 1]   # U'
    A[1, 3] = Gi[1, 0];      A[1, 4] = Gi[1, 1]                       # V'
    A[2, 0] = p * C13 / C33; A[2, 1] = p * C36 / C33; A[2, 5] = 1.0 / C33   # W'
    A[3, 0] = p * p * Q11;   A[3, 1] = p * p * Q16;   A[3, 5] = -p * C13 / C33  # X'
    A[4, 0] = p * p * Q16;   A[4, 1] = p * p * Q66;   A[4, 5] = -p * C36 / C33  # Y'
    A[5, 3] = p                                                        # Z'
    return A, (C11, C12, C13, C16, C22, C23, C26, C33, C36)


class ExactCyl:
    """Exact cylindrical-bending solution.  Layer 0 = bottom face."""

    def __init__(self, thick, angles_deg, mat_names, material_db, L, q0=1.0, n=1):
        self.thick = np.asarray(thick, float)
        self.h = float(self.thick.sum())
        self.L = float(L)
        self.q0 = float(q0)
        self.p = n * np.pi / self.L
        self.C = [rotated_stiffness_6x6(material_db[m]['E'], material_db[m]['G'],
                                        material_db[m]['nu'], a)
                  for m, a in zip(mat_names, angles_deg)]
        self.z_bot = np.concatenate([[0.0], np.cumsum(self.thick)])

        self._A = []
        self._con = []
        self._T = []                      # cumulative transfer from z=0 to bottom of layer k
        T = np.eye(6)
        for k in range(len(self.thick)):
            Ak, ck = _layer_A(self.C[k], self.p)
            self._A.append(Ak)
            self._con.append(ck)
            self._T.append(T.copy())
            T = expm(Ak * self.thick[k]) @ T
        self.T_tot = T

        # BCs: bottom  X=Y=Z=0  ->  s0 = [U0, V0, W0, 0, 0, 0]
        #      top     X=Y=0, Z=q0
        M = self.T_tot[3:6, 0:3]
        rhs = np.array([0.0, 0.0, self.q0])
        uvw0 = np.linalg.solve(M, rhs)
        self.s0 = np.array([uvw0[0], uvw0[1], uvw0[2], 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ state
    def _layer_of(self, z):
        k = int(np.searchsorted(self.z_bot[1:-1], z, side='left'))
        return int(np.clip(k, 0, len(self.thick) - 1))

    def state(self, z, layer=None):
        """s(z) = [U, V, W, X, Y, Z] with z measured from the BOTTOM face."""
        k = self._layer_of(z) if layer is None else layer
        return expm(self._A[k] * (z - self.z_bot[k])) @ (self._T[k] @ self.s0)

    # ----------------------------------------------------------------- stress
    def stress_amp(self, z, layer=None):
        """Amplitudes [S11, S22, S33, S23, S13, S12] (OpenSG Voigt order).

        sigma_{11,22,33,12} multiply sin(px); sigma_{13,23} multiply cos(px).
        """
        k = self._layer_of(z) if layer is None else layer
        U, V, W, X, Y, Z = self.state(z, layer=k)
        C11, C12, C13, C16, C22, C23, C26, C33, C36 = self._con[k]
        p = self.p
        dW = (p * C13 * U + p * C36 * V + Z) / C33            # eps33
        S11 = -p * C11 * U - p * C16 * V + C13 * dW
        S22 = -p * C12 * U - p * C26 * V + C23 * dW
        S12 = -p * C16 * U - p * C66_of(self.C[k]) * V + C36 * dW
        return np.array([S11, S22, Z, Y, X, S12])

    def strain_amp(self, z, layer=None):
        """Amplitudes [e11, e22, e33, g23, g13, g12] (same sin/cos split as stress)."""
        k = self._layer_of(z) if layer is None else layer
        U, V, W, X, Y, Z = self.state(z, layer=k)
        C = self.C[k]
        C13, C33, C36 = C[0, 2], C[2, 2], C[2, 5]
        p = self.p
        e33 = (p * C13 * U + p * C36 * V + Z) / C33
        Gs = np.array([[C[4, 4], C[3, 4]], [C[3, 4], C[3, 3]]])
        g13, g23 = np.linalg.solve(Gs, np.array([X, Y]))
        return np.array([-p * U, 0.0, e33, g23, g13, -p * V])

    def disp_amp(self, z, layer=None):
        """[U, V, W]; u,v multiply cos(px), w multiplies sin(px)."""
        return self.state(z, layer=layer)[:3]

    # ----------------------------------------------------------- sampling grid
    def profile(self, n_per_layer=41, eps=1e-12):
        """Through-thickness sample honouring the material discontinuities.

        Returns (zc, sig, eps6, uvw) with zc measured from the MID-surface and both
        one-sided limits present at every interface.
        """
        zs, lay = [], []
        for k in range(len(self.thick)):
            a, b = self.z_bot[k], self.z_bot[k + 1]
            zz = np.linspace(a + eps * (b - a), b - eps * (b - a), n_per_layer)
            zs.append(zz)
            lay.append(np.full(n_per_layer, k))
        zs = np.concatenate(zs)
        lay = np.concatenate(lay)
        sig = np.array([self.stress_amp(z, layer=k) for z, k in zip(zs, lay)])
        eps6 = np.array([self.strain_amp(z, layer=k) for z, k in zip(zs, lay)])
        uvw = np.array([self.disp_amp(z, layer=k) for z, k in zip(zs, lay)])
        return zs - self.h / 2.0, sig, eps6, uvw

    # --------------------------------------------------------------- residuals
    def bc_residual(self):
        """max |traction BC violation| normalised by q0 (should be ~1e-12)."""
        sb = self.state(0.0, layer=0)
        st = self.state(self.h, layer=len(self.thick) - 1)
        return max(abs(sb[3]), abs(sb[4]), abs(sb[5]),
                   abs(st[3]), abs(st[4]), abs(st[5] - self.q0)) / abs(self.q0)


def C66_of(C):
    return C[5, 5]
