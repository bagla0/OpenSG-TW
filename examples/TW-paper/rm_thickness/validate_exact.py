"""validate_exact.py -- sanity checks on the exact cylindrical-bending solver.

Checks that do NOT depend on any published table:
  (1) traction BCs on both faces (machine precision by construction);
  (2) global equilibrium: N11 = 0, M11 = q0/p^2, Q1 = q0/p, obtained by integrating the
      recovered 3-D stress -- these are exact consequences of the 3-D equations and tie
      the elasticity solution to the plate resultants the RM model must reproduce;
  (3) thin-limit: sigma11 -> CLT distribution as L/h -> inf;
  (4) isotropic thick plate: sigma13 within a few % of the parabolic 1.5*Q/h profile.
"""
import numpy as np

from exact_cyl import ExactCyl
from materials import MATDB


def resultants(sol, n_gauss=200):
    """[N11, M11, Q1] from the exact through-thickness stress."""
    xi, wq = np.polynomial.legendre.leggauss(n_gauss)
    N11 = M11 = Q1 = 0.0
    for k in range(len(sol.thick)):
        a, b = sol.z_bot[k], sol.z_bot[k + 1]
        zq = 0.5 * (a + b) + 0.5 * (b - a) * xi
        wt = 0.5 * (b - a) * wq
        for z, w in zip(zq, wt):
            s = sol.stress_amp(z, layer=k)
            zc = z - sol.h / 2.0
            N11 += w * s[0]
            M11 += w * s[0] * zc
            Q1 += w * s[4]
    return N11, M11, Q1


def main():
    thick = [1.0 / 3, 1.0 / 3, 1.0 / 3]
    ang = [0.0, 90.0, 0.0]
    mats = ['pagano'] * 3
    h = 1.0
    q0 = 1.0

    print("=" * 78)
    print("[0/90/0] Pagano material, h = 1")
    print("=" * 78)
    print(f"{'S=L/h':>7} {'BC resid':>11} {'N11/(q0 h)':>13} {'M11/(q0/p^2)':>14} "
          f"{'Q1/(q0/p)':>12}")
    for S in [4, 10, 20, 50, 100]:
        L = S * h
        sol = ExactCyl(thick, ang, mats, MATDB, L, q0=q0)
        N11, M11, Q1 = resultants(sol)
        p = sol.p
        print(f"{S:>7} {sol.bc_residual():>11.2e} {N11 / (q0 * h):>13.2e} "
              f"{M11 / (q0 / p ** 2):>14.9f} {Q1 / (q0 / p):>12.9f}")

    # ---- isotropic thick plate: sigma13 vs parabola --------------------------
    print()
    print("=" * 78)
    print("isotropic single layer, S = 10 : sigma13 vs 1.5*Q1/h parabola")
    print("=" * 78)
    iso = {'iso': {'E': [70e9] * 3, 'G': [70e9 / 2.6] * 3, 'nu': [0.3] * 3, 'rho': 1.0}}
    sol = ExactCyl([1.0], [0.0], ['iso'], iso, 10.0, q0=1.0)
    zc, sig, _, _ = sol.profile(n_per_layer=41)
    Q1 = 1.0 / sol.p
    par = 1.5 * Q1 / sol.h * (1 - 4 * zc ** 2 / sol.h ** 2)
    err = np.max(np.abs(sig[:, 4] - par)) / np.max(np.abs(par))
    print(f"  max |sigma13 - parabola| / max|parabola| = {err:.3e}")
    print(f"  sigma13 at mid-plane: exact {sig[len(zc)//2, 4]:.6f}  "
          f"parabola {1.5*Q1/sol.h:.6f}")

    # ---- thin limit: sigma11 vs CLT -----------------------------------------
    print()
    print("=" * 78)
    print("thin limit [0/90/0]: sigma11(z) vs CLT  (relative L2 over the thickness)")
    print("=" * 78)
    from msg_rm_plate import rm_plate_msg
    for S in [4, 10, 20, 50, 100, 400]:
        L = S * h
        sol = ExactCyl(thick, ang, mats, MATDB, L, q0=q0)
        zc, sig, _, _ = sol.profile(n_per_layer=41)
        obj = rm_plate_msg(thick, ang, mats, MATDB, n_per_layer=4, elem_order=3, z_ref=h / 2)
        A6 = obj['A6']
        ix = np.array([0, 2, 3, 5])
        Ar = A6[np.ix_(ix, ix)]
        F = np.array([0.0, 0.0, q0 / sol.p ** 2, 0.0])
        Er = np.linalg.solve(Ar, F)
        E6 = np.zeros(6)
        E6[ix] = Er
        clt = []
        for z in zc:
            k = obj['elem_layer'][int(np.clip(np.searchsorted(
                obj['node_x'][::obj['elem_order']][1:], z, side='right'), 0,
                len(obj['elem_layer']) - 1))]
            g = np.zeros(6)
            g[0] = E6[0] + z * E6[3]
            g[1] = E6[1] + z * E6[4]
            g[5] = E6[2] + z * E6[5]
            # plane-stress (sigma33 = sigma13 = sigma23 = 0) reduction, i.e. classical
            C = obj['C_layers'][k]
            keep = np.array([0, 1, 5])
            drop = np.array([2, 3, 4])
            Cr = C[np.ix_(keep, keep)] - C[np.ix_(keep, drop)] @ np.linalg.solve(
                C[np.ix_(drop, drop)], C[np.ix_(drop, keep)])
            clt.append((Cr @ g[keep])[0])
        clt = np.array(clt)
        rel = np.linalg.norm(sig[:, 0] - clt) / np.linalg.norm(clt)
        print(f"  S = {S:>4} :  ||sigma11_exact - sigma11_CLT|| / ||CLT|| = {rel:.3e}")


if __name__ == '__main__':
    main()
