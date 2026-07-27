"""tsdt.py -- Reddy's third-order shear deformation theory (TSDT, JAM 51 (1984) 745)
on the cylindrical-bending benchmark, against exact elasticity and OpenSG-RM.

Kinematics (symmetric bending, single Navier harmonic, cross-ply/sandwich):

    u(x,z) = z*phi(x) - c1 z^3 (phi + w,x),   w(x,z) = w0(x),   c1 = 4/(3 h^2)

so gamma_xz = (1 - 3 c1 z^2)(phi + w,x) is parabolic and vanishes on the faces by
construction -- the TSDT selling point: no shear correction factor.  The price is that
the CONSTITUTIVE transverse shear C55(z) * gamma(z) is still discontinuous at ply
interfaces (C55 jumps, gamma is continuous), and for a soft core the single smooth
parabola cannot follow the true near-constant-core / parabolic-face flow.

Same conventions as models.py: load sigma_33(top) = q0 sin(px); in-plane modulus is
the eps22 = 0, sigma33-condensed Qb11 = C11 - C13^2/C33 (identical to the exact
solver), so every model in the comparison shares its material reduction.

Outputs: results/table_tsdt.csv + printed table + the profile figures used as
Example 4 in the paper.
"""
import os

import numpy as np

from jaxcfg import jnp                       # noqa: F401  (x64 first)
import sg_plate as SG
import models as M
from materials import MATDB, layer_stiffness
from exact_cyl import ExactCyl

HERE = os.path.dirname(os.path.abspath(__file__))


def tsdt_solve(thick, angles, mats, matdb, S, q0=1.0, n_out=61):
    """TSDT Navier solution + recovery for cylindrical bending.  Amplitudes."""
    thick = np.asarray(thick, float)
    h = float(thick.sum())
    L = S * h
    p = np.pi / L
    c1 = 4.0 / (3.0 * h * h)
    C = np.asarray(layer_stiffness(mats, angles, matdb))
    zb = np.concatenate([[0.0], np.cumsum(thick)]) - h / 2.0

    # quadrature per layer for the stiffness integrals
    xi, wq = np.polynomial.legendre.leggauss(8)
    K = np.zeros((2, 2))                      # dofs d = [W, P]
    for k in range(thick.size):
        a, b = zb[k], zb[k + 1]
        zq = 0.5 * (a + b) + 0.5 * (b - a) * xi
        wt = 0.5 * (b - a) * wq
        Qb = C[k, 0, 0] - C[k, 0, 2] ** 2 / C[k, 2, 2]
        G = C[k, 4, 4]
        # eps_x amplitude = aP(z)*P + aW(z)*W ;  gamma amplitude = g(z)*(P + p W)
        aP = -p * (zq - c1 * zq ** 3)
        aW = c1 * zq ** 3 * p * p
        g = 1.0 - 3.0 * c1 * zq ** 2
        for z_, w_, ap_, aw_, g_ in zip(zq, wt, aP, aW, g):
            e = np.array([aw_, ap_])          # [dW, dP] coefficients of eps_x
            gv = np.array([p * g_, g_])       # coefficients of gamma_xz
            K += w_ * (Qb * np.outer(e, e) + G * np.outer(gv, gv))
    d = np.linalg.solve(K, np.array([q0, 0.0]))
    W, P = d

    # ---- through-thickness recovery on the standard interface-doubled grid ----
    zs, lay = [], []
    for k in range(thick.size):
        a, b = zb[k], zb[k + 1]
        zz = np.linspace(a + 1e-9 * (b - a), b - 1e-9 * (b - a), n_out)
        zs.append(zz); lay.append(np.full(n_out, k))
    z = np.concatenate(zs); lay = np.concatenate(lay)

    Qb = C[lay, 0, 0] - C[lay, 0, 2] ** 2 / C[lay, 2, 2]
    G = C[lay, 4, 4]
    eps = (-p * (z - c1 * z ** 3)) * P + (c1 * z ** 3 * p * p) * W
    s11 = Qb * eps                                           # sin family
    txz_con = G * (1.0 - 3.0 * c1 * z ** 2) * (P + p * W)    # constitutive, cos
    # equilibrium recovery: txz' = -s11,x -> amplitude' = -p*s11
    txz_eq = np.zeros_like(z)
    txz_eq[1:] = np.cumsum(0.5 * (-p) * (s11[1:] + s11[:-1]) * np.diff(z))
    s33 = np.zeros_like(z)
    s33[1:] = np.cumsum(0.5 * p * (txz_eq[1:] + txz_eq[:-1]) * np.diff(z))
    u = (z - c1 * z ** 3) * P - c1 * z ** 3 * p * W          # cos family
    return dict(z=z, d=d, W=W, s11=s11, txz_con=txz_con, txz_eq=txz_eq,
                s33=s33, u=u, p=p, h=h)


def run_case(name, thick, angles, mats, S, rows, npl_sg=6):
    h = float(np.sum(thick))
    ex = ExactCyl(thick, angles, mats, MATDB, S * h)
    zc, sig_e, _, uvw_e = ex.profile(n_per_layer=61)
    ts = tsdt_solve(thick, angles, mats, MATDB, S, n_out=61)
    r = M.run(thick, angles, mats, MATDB, S, n_per_layer_out=61, npl_sg=npl_sg)

    i0 = int(np.argmin(np.abs(zc)))
    w_ex = uvw_e[i0, 2]
    e = dict(
        s11_tsdt=M.relerr(ts['s11'], sig_e[:, 0]),
        s11_msg=M.relerr(r['msg']['s11'], sig_e[:, 0]),
        txz_fsdt=M.relerr(r['fsdt']['s13'], sig_e[:, 4]),
        txz_tsdt_c=M.relerr(ts['txz_con'], sig_e[:, 4]),
        txz_tsdt_e=M.relerr(ts['txz_eq'], sig_e[:, 4]),
        txz_msg=M.relerr(r['msg']['s13'], sig_e[:, 4]),
        s33_tsdt=M.relerr(ts['s33'], sig_e[:, 2]),
        s33_msg=M.relerr(r['msg']['s33'], sig_e[:, 2]),
        w_tsdt=abs(ts['W'] / w_ex - 1.0),
        w_fsdt=abs(M.plate_strains(r['sg']['A6'], ex.p)[0] * 0.0
                   + _w_navier(r, ex.p, which='fsdt') / w_ex - 1.0),
        w_msg=abs(_w_navier(r, ex.p, which='msg') / w_ex - 1.0),
    )
    rows.append(dict(case=name.strip(), S=S, **e))
    print(f"{name} S={S:>4} | txz: FSDT {100*e['txz_fsdt']:6.2f}%  "
          f"TSDT-con {100*e['txz_tsdt_c']:6.2f}%  TSDT-eq {100*e['txz_tsdt_e']:6.2f}%  "
          f"MSG {100*e['txz_msg']:6.2f}% | s33: TSDT {100*e['s33_tsdt']:5.2f}% "
          f"MSG {100*e['s33_msg']:5.2f}% | w err: TSDT {100*e['w_tsdt']:5.2f}% "
          f"FSDT {100*e['w_fsdt']:5.2f}% MSG {100*e['w_msg']:5.2f}%")
    return dict(zc=zc, sig_e=sig_e, ts=ts, r=r, h=h)


def _w_navier(r, p, which):
    """Central deflection of the FSDT / RM plate for the cylindrical problem.

    Statically determinate: kappa = M A^-1 rows; w'' = -kappa - gamma' ... for the
    single harmonic:  W = (kappa + p*gamma) / p^2  with kappa the bending strain from
    M11 = q0/p^2 and gamma = Q1/G = (q0/p)/G.
    """
    sg = r['sg']
    A6 = np.asarray(sg['A6'])
    ix = np.array([0, 2, 3, 5])
    Er = np.linalg.solve(A6[np.ix_(ix, ix)], np.array([0., 0., 1.0 / p ** 2, 0.]))
    kap = Er[2]
    if which == 'fsdt':
        Gs = np.asarray(M.fsdt_shear(sg)[0])
    else:
        Gs = np.asarray(sg['G_msg'])
    gam = np.linalg.solve(Gs, np.array([1.0 / p, 0.0]))[0]
    return kap / p ** 2 + gam / p


def figures(res_cross, res_sand, S):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT = os.path.join(HERE, 'figures')
    for tag, R in (('crossply', res_cross), ('sandwich', res_sand)):
        zb = R['zc'] / R['h']
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
        ax = axes[0]
        ax.plot(R['sig_e'][:, 4], zb, 'k', lw=2.2, label='Analytic 3-D')
        ax.plot(R['ts']['txz_con'], zb, '#2ca02c', lw=1.6, ls=':',
                label='TSDT constitutive')
        ax.plot(R['ts']['txz_eq'], zb, '#d62728', lw=1.6, ls='--',
                label='TSDT equilibrium')
        ax.plot(R['r']['msg']['s13'], zb, '#1f77b4', lw=1.6, ls='-.',
                label='OpenSG-RM')
        ax.set_xlabel(r'$\sigma_{13}/q_0$')
        ax.set_ylabel(r'$z/h$'); ax.set_ylim(-0.5, 0.5); ax.grid(alpha=0.25, lw=0.5)
        ax = axes[1]
        ax.plot(R['sig_e'][:, 2], zb, 'k', lw=2.2, label='Analytic 3-D')
        ax.plot(R['ts']['s33'], zb, '#d62728', lw=1.6, ls='--',
                label='TSDT equilibrium')
        ax.plot(R['r']['msg']['s33'], zb, '#1f77b4', lw=1.6, ls='-.',
                label='OpenSG-RM')
        ax.set_xlabel(r'$\sigma_{33}/q_0$')
        ax.set_ylabel(r'$z/h$'); ax.set_ylim(-0.5, 0.5); ax.grid(alpha=0.25, lw=0.5)
        h_, l_ = axes[0].get_legend_handles_labels()
        fig.legend(h_, l_, loc='center left', bbox_to_anchor=(1.0, 0.5),
                   frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'tsdt_{tag}_S{S}.png'), dpi=180,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote tsdt_{tag}_S{S}.png")


def main():
    rows = []
    t3 = [1 / 3] * 3
    ts_ = [0.1, 0.8, 0.1]
    for S in (100, 10, 4):
        rc = run_case("[0/90/0] ", t3, [0., 90., 0.], ['pagano'] * 3, S, rows)
        rs = run_case("[0/core/0]", ts_, [0.] * 3, ['face', 'core', 'face'], S,
                      rows, npl_sg=8)
        if S == 10:
            figures(rc, rs, S)
    import csv
    path = os.path.join(HERE, 'results', 'table_tsdt.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
