"""sweep.py -- the whole Garg-2023 training box, in one vmapped call.

Garg et al. draw 500 Sobol samples from the box of Table 1, run FSDT and Pagano's 3-D
elasticity solution for each, and fit a Gaussian process to the DIFFERENCE.  Their
corrected FSDT is then accurate inside that box, by construction.

Here the same box is sampled and the same two references are computed -- but the MSG
model is never shown the elasticity answer.  Every laminate is an out-of-sample
prediction, because there is no sample.  ``jax.vmap`` maps the SG solve, the recovery and
the exact solution over the whole population at once.

Table 1 of Garg et al. (per ply):

    E11   20 - 300 GPa        G23      0.4 - 7 GPa      theta   -90 - 90 deg
    E33    1 - 10.5 GPa       nu12=nu13 0.22 - 0.28     t_ply   0.01 - 0.4
    G12 = G13  0.5 - 8 GPa    nu23     0.25 - 0.5       l/h        4 - 100

(E22 is not listed; transverse isotropy in the 2-3 plane, E22 = E33, is assumed.)
"""
import argparse
import os
import time

import numpy as np
from scipy.stats import qmc

from jaxcfg import jax, jnp
from materials import build_stiffness_6x6, rotation_6x6
from sg_plate import sg_plate_solve
from batched import sample_spec, sg_recover, exact_profile
from models import IX_CYL

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)

BOUNDS = dict(E11=(20e9, 300e9), E33=(1e9, 10.5e9), G12=(0.5e9, 8e9),
              G23=(0.4e9, 7e9), nu12=(0.22, 0.28), nu23=(0.25, 0.5),
              theta=(-90.0, 90.0), t=(0.01, 0.4))
S_BOUNDS = (4.0, 100.0)
NVAR = 8            # per ply


def draw(n_samples, nlay, seed=0):
    """Sobol sample of the Garg box -> (thick, C_layers, S), plus a validity mask."""
    dim = nlay * NVAR + 1
    s = qmc.Sobol(d=dim, scramble=True, seed=seed).random(n_samples)
    keys = list(BOUNDS)
    lo = np.array([BOUNDS[k][0] for k in keys])
    hi = np.array([BOUNDS[k][1] for k in keys])
    ply = s[:, :nlay * NVAR].reshape(n_samples, nlay, NVAR) * (hi - lo) + lo
    S = s[:, -1] * (S_BOUNDS[1] - S_BOUNDS[0]) + S_BOUNDS[0]

    thick = ply[:, :, keys.index('t')]
    C = np.empty((n_samples, nlay, 6, 6))
    ok = np.ones(n_samples, bool)
    for i in range(n_samples):
        for k in range(nlay):
            v = dict(zip(keys, ply[i, k]))
            E = [v['E11'], v['E33'], v['E33']]
            G = [v['G12'], v['G12'], v['G23']]
            nu = [v['nu12'], v['nu12'], v['nu23']]
            Ck = build_stiffness_6x6(E, G, nu)
            if np.min(np.linalg.eigvalsh(Ck)) <= 0:
                ok[i] = False
            R = rotation_6x6(v['theta'])
            C[i, k] = R @ Ck @ R.T
    return thick, C, S, ok


# --------------------------------------------------------------------- one case
def _case(thick, C, S, spec, n_per_layer, p, ks):
    h = jnp.sum(thick)
    z_ref = 0.5 * h
    sol = sg_plate_solve(thick, C, n_per_layer, p, z_ref)
    sol = {**sol, 'C_layers': C}

    pp = jnp.pi / (S * h)
    A6 = sol['A6']
    ix = jnp.asarray(IX_CYL)
    Er = jnp.linalg.solve(A6[jnp.ix_(ix, ix)],
                          jnp.array([0.0, 0.0, 1.0 / pp ** 2, 0.0]))
    E6 = jnp.zeros(6).at[ix].set(Er)
    Z6 = jnp.zeros(6)

    _, Sig_m = sg_recover(sol, thick, z_ref, spec, E6, Z6, Z6)
    _, Sig_s = sg_recover(sol, thick, z_ref, spec, Z6, pp * E6, Z6)

    z = (jnp.concatenate([jnp.zeros(1), jnp.cumsum(thick)])[:-1][spec['lay']]
         + spec['fr'] * thick[spec['lay']] - z_ref)
    s13 = Sig_s[:, 4]
    trap = 0.5 * (s13[1:] + s13[:-1]) * jnp.diff(z)
    s33 = jnp.concatenate([jnp.zeros(1), jnp.cumsum(pp * trap)])

    # FSDT: single director + shear-correction factor
    gk = jnp.stack([jnp.array([[c[4, 4], c[4, 3]], [c[3, 4], c[3, 3]]]) for c in C])
    Gs = ks * jnp.einsum('k,kab->ab', thick, gk)
    gam = jnp.linalg.solve(Gs, jnp.array([1.0 / pp, 0.0]))
    s13_f = jnp.einsum('nab,b->na', gk[spec['lay']], gam)[:, 0]

    sig_e, _, _, bc = exact_profile(thick, C, S, spec)

    def rel(a, b):
        return jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-300)

    return jnp.array([rel(s13_f, sig_e[:, 4]), rel(s13, sig_e[:, 4]),
                      rel(s33, sig_e[:, 2]), rel(Sig_m[:, 0], sig_e[:, 0]),
                      jnp.abs(s33[-1] - 1.0), bc])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--n-samples', type=int, default=512)
    ap.add_argument('-l', '--n-layers', type=int, default=8)
    ap.add_argument('--npl', type=int, default=3, help='SG elements per ply')
    ap.add_argument('--order', type=int, default=3)
    ap.add_argument('--n-out', type=int, default=15, help='sample points per ply')
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    print(f"Sobol sample of the Garg-2023 box: {args.n_samples} laminates "
          f"x {args.n_layers} plies")
    thick, C, S, ok = draw(args.n_samples, args.n_layers)
    print(f"  {int(ok.sum())}/{args.n_samples} admissible (positive-definite C)")
    thick, C, S = thick[ok], C[ok], S[ok]

    spec = sample_spec(args.n_layers, args.npl, args.order, args.n_out)
    f = jax.jit(jax.vmap(lambda t, c, s: _case(t, c, s, spec, args.npl, args.order,
                                               5.0 / 6.0)))

    t0 = time.time()
    out = []
    for i in range(0, thick.shape[0], args.batch):
        sl = slice(i, i + args.batch)
        out.append(np.asarray(f(jnp.asarray(thick[sl]), jnp.asarray(C[sl]),
                                jnp.asarray(S[sl]))))
        print(f"  {min(i + args.batch, thick.shape[0]):>5}/{thick.shape[0]}"
              f"   {time.time() - t0:6.1f} s", end='\r')
    err = np.concatenate(out)
    dt = time.time() - t0
    print(f"\n  {err.shape[0]} laminates in {dt:.1f} s "
          f"({1e3 * dt / err.shape[0]:.1f} ms each, CPU)")

    good = np.isfinite(err).all(axis=1)
    err = err[good]
    S = S[good]
    names = ['sigma13 FSDT', 'sigma13 MSG', 'sigma33 MSG', 'sigma11 MSG',
             'sigma33 top-face closure', 'EXACT traction-BC residual']
    bc_max = float(np.max(err[:, 5]))
    print(f"\n  oracle integrity: worst exact-solution traction-BC residual "
          f"= {bc_max:.2e}" + ("   ok" if bc_max < 1e-8 else "   *** SUSPECT ***"))
    print(f"\n  relative L2 error vs exact 3-D elasticity, over {err.shape[0]} laminates")
    print(f"  {'quantity':<26} {'median':>9} {'90th pct':>9} {'max':>9}")
    for j, nm in enumerate(names):
        e = 100 * err[:, j]
        print(f"  {nm:<26} {np.median(e):>8.2f}% {np.percentile(e, 90):>8.2f}%"
              f" {np.max(e):>8.2f}%")

    thin = S >= 20
    print(f"\n  restricted to S >= 20 ({int(thin.sum())} laminates)")
    for j, nm in enumerate(names[:4]):
        e = 100 * err[thin, j]
        print(f"  {nm:<26} {np.median(e):>8.2f}% {np.percentile(e, 90):>8.2f}%"
              f" {np.max(e):>8.2f}%")

    np.savez(os.path.join(OUT, 'sweep.npz'), err=err, S=S, names=np.array(names))
    print(f"\nwrote {os.path.join(OUT, 'sweep.npz')}")


if __name__ == '__main__':
    main()
