"""pilot.py -- does MSG-VAM reproduce the 3-D elasticity through-thickness distributions
that Garg et al. (2023) obtain by training a GPR surrogate on those same solutions?

Cases mirror Garg et al. sec. 3:
  A  [0/90/0] and [90/0/90], Pagano material, S = 100 and S = 5
  B  4-layer [0/90/90/0] and [0/90/0/90], AS4-type material
  C  sandwich [0/core/0], 0.1h / 0.8h / 0.1h  (the case where FSDT + shear-correction
     factors are known to fail -- cf. the Ansys SCF study)

Metric: relative L2 error of the through-thickness profile against the exact 3-D
elasticity solution, per model.
"""
import numpy as np

from exact_cyl import ExactCyl
from materials import MATDB
import cyl_models as CM


def relerr(a, b):
    b = np.asarray(b, float)
    return float(np.linalg.norm(np.asarray(a, float) - b) / (np.linalg.norm(b) + 1e-300))


def run_case(name, thick, angles, mats, S, q0=1.0, n_per_layer=61, npl_sg=4, order=3):
    thick = np.asarray(thick, float)
    h = float(thick.sum())
    L = S * h
    ex = ExactCyl(thick, angles, mats, MATDB, L, q0=q0)
    p = ex.p

    obj = CM.build(thick, angles, mats, MATDB, n_per_layer=npl_sg, elem_order=order)
    E6 = CM.plate_strains(obj['A6'], p, q0=q0)

    zc, sig_e, _, _ = ex.profile(n_per_layer=n_per_layer)
    fs = CM.fsdt_profile(obj, E6, p, q0=q0, n_per_layer=n_per_layer)
    cl = CM.clt_equil_profile(obj, E6, p, n_per_layer=n_per_layer)
    mg = CM.msg_profile(obj, E6, p, n_per_layer=n_per_layer)

    assert np.allclose(zc, fs['z'], atol=1e-9 * h), "z grids differ"

    rows = []
    rows.append(('sigma11', relerr(fs['s11'], sig_e[:, 0]), relerr(cl['s11'], sig_e[:, 0]),
                 relerr(mg['s11'], sig_e[:, 0])))
    rows.append(('sigma13', relerr(fs['s13'], sig_e[:, 4]), relerr(cl['s13'], sig_e[:, 4]),
                 relerr(mg['s13'], sig_e[:, 4])))
    rows.append(('sigma33', np.nan, relerr(cl['s33'], sig_e[:, 2]),
                 relerr(mg['s33'], sig_e[:, 2])))

    print(f"\n{name}   S = L/h = {S}")
    print(f"   {'quantity':<9} {'FSDT':>12} {'CLT+equil':>12} {'MSG-VAM':>12}")
    for q, a, b, c in rows:
        sa = '   n/a      ' if np.isnan(a) else f"{100*a:>11.3f}%"
        print(f"   {q:<9} {sa} {100*b:>11.3f}% {100*c:>11.3f}%")

    # peak values for eyeballing against the published figures
    i_s13 = np.argmax(np.abs(sig_e[:, 4]))
    i_s33 = np.argmax(np.abs(sig_e[:, 2]))
    print(f"   peak sigma13/q0 : exact {sig_e[i_s13,4]:>10.4f} | FSDT {fs['s13'][i_s13]:>10.4f}"
          f" | CLT {cl['s13'][i_s13]:>10.4f} | MSG {mg['s13'][i_s13]:>10.4f}")
    print(f"   peak sigma33/q0 : exact {sig_e[i_s33,2]:>10.4f} | FSDT      n/a  "
          f" | CLT {cl['s33'][i_s33]:>10.4f} | MSG {mg['s33'][i_s33]:>10.4f}")
    return dict(name=name, S=S, zc=zc, exact=sig_e, fsdt=fs, clt=cl, msg=mg, obj=obj,
                E6=E6, p=p, h=h)


def main():
    t3 = [1 / 3, 1 / 3, 1 / 3]
    print("=" * 78)
    print("A. three-layer cross-ply, Pagano material  (Garg figs 3-4)")
    print("=" * 78)
    for S in (100, 10, 5, 4):
        run_case("[0/90/0] ", t3, [0., 90., 0.], ['pagano'] * 3, S)
    for S in (100, 5):
        run_case("[90/0/90]", t3, [90., 0., 90.], ['pagano'] * 3, S)

    print()
    print("=" * 78)
    print("B. four-layer AS4  (Garg fig 6)")
    print("=" * 78)
    t4 = [0.25] * 4
    for S in (10, 5):
        run_case("[0/90/90/0]", t4, [0., 90., 90., 0.], ['as4'] * 4, S)
        run_case("[0/90/0/90]", t4, [0., 90., 0., 90.], ['as4'] * 4, S)

    print()
    print("=" * 78)
    print("C. sandwich [0/core/0], 0.1h/0.8h/0.1h  (Garg fig 7)")
    print("=" * 78)
    ts = [0.1, 0.8, 0.1]
    for S in (20, 10, 5, 4):
        run_case("[0/core/0]", ts, [0., 0., 0.], ['face', 'core', 'face'], S,
                 npl_sg=6)

    print()
    print("=" * 78)
    print("D. angle-ply -- OUTSIDE Pagano's original construction, exact here")
    print("=" * 78)
    for th in (15., 30., 45.):
        run_case(f"[0/{th:g}/0]", t3, [0., th, 0.], ['pagano'] * 3, 10)


if __name__ == '__main__':
    main()
