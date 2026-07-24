"""run_study.py -- the case-by-case comparison table.

Cases mirror Garg et al. (2023) sec. 3:
  A  three-layer cross-ply [0/90/0] and [90/0/90], Pagano material   (their figs 3, 4)
  B  four-layer [0/90/90/0] and [0/90/0/90], AS4-type material       (their fig 6)
  C  sandwich [0/core/0] at 0.1h / 0.8h / 0.1h                       (their fig 7)
  D  angle-ply [0/theta/0] -- outside Pagano's own construction, exact here

Writes ``results/table.csv`` and prints the table.
"""
import os

import numpy as np

from jaxcfg import jnp          # noqa: F401  (enables x64 before anything else)
import models as M
from materials import MATDB

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)

HDR = (f"   {'quantity':<9} {'FSDT':>11} {'CLT+equil':>11} {'MSG-VAM':>11}")


def one(name, thick, angles, mats, S, rows, npl_sg=6):
    r = M.run(thick, angles, mats, MATDB, S, npl_sg=npl_sg)
    ex = r['exact']
    e = {}
    e['sigma11'] = (M.relerr(r['fsdt']['s11'], ex[:, 0]),
                    M.relerr(r['clt']['s11'], ex[:, 0]),
                    M.relerr(r['msg']['s11'], ex[:, 0]))
    e['sigma13'] = (M.relerr(r['fsdt']['s13'], ex[:, 4]),
                    M.relerr(r['clt']['s13'], ex[:, 4]),
                    M.relerr(r['msg']['s13'], ex[:, 4]))
    e['sigma33'] = (np.nan,
                    M.relerr(r['clt']['s33'], ex[:, 2]),
                    M.relerr(r['msg']['s33'], ex[:, 2]))
    print(f"\n{name}   S = L/h = {S}")
    print(HDR)
    for q in ('sigma11', 'sigma13', 'sigma33'):
        a, b, c = e[q]
        sa = '    n/a    ' if np.isnan(a) else f"{100 * a:>10.3f}%"
        print(f"   {q:<9} {sa} {100 * b:>10.3f}% {100 * c:>10.3f}%")
        rows.append(dict(case=name.strip(), S=S, quantity=q,
                         fsdt=a, clt_equil=b, msg=c))
    i13 = int(np.argmax(np.abs(ex[:, 4])))
    print(f"   peak sigma13/q0 : exact {ex[i13, 4]:>9.4f} |"
          f" FSDT {r['fsdt']['s13'][i13]:>9.4f} | MSG {r['msg']['s13'][i13]:>9.4f}")
    return r


def main():
    rows = []
    t3 = [1 / 3] * 3
    print("=" * 72)
    print("A. three-layer cross-ply, Pagano material")
    print("=" * 72)
    for S in (100, 20, 10, 5, 4):
        one("[0/90/0] ", t3, [0., 90., 0.], ['pagano'] * 3, S, rows)
    for S in (100, 5):
        one("[90/0/90]", t3, [90., 0., 90.], ['pagano'] * 3, S, rows)

    print("\n" + "=" * 72)
    print("B. four-layer AS4")
    print("=" * 72)
    t4 = [0.25] * 4
    for S in (10, 5):
        one("[0/90/90/0]", t4, [0., 90., 90., 0.], ['as4'] * 4, S, rows)
        one("[0/90/0/90]", t4, [0., 90., 0., 90.], ['as4'] * 4, S, rows)

    print("\n" + "=" * 72)
    print("C. sandwich [0/core/0], 0.1h / 0.8h / 0.1h")
    print("=" * 72)
    ts = [0.1, 0.8, 0.1]
    for S in (20, 10, 5, 4):
        one("[0/core/0]", ts, [0.] * 3, ['face', 'core', 'face'], S, rows, npl_sg=8)

    print("\n" + "=" * 72)
    print("D. angle-ply (outside Pagano's original construction)")
    print("=" * 72)
    for th in (15., 30., 45.):
        one(f"[0/{th:g}/0]", t3, [0., th, 0.], ['pagano'] * 3, 10, rows)

    import csv
    path = os.path.join(OUT, 'table.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['case', 'S', 'quantity', 'fsdt',
                                          'clt_equil', 'msg'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == '__main__':
    main()
