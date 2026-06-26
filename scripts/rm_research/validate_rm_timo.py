"""Validate the RM Timoshenko (V1) 6x6 on the isotropic tube (Table 3.1)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from msg_rm_timo import timoshenko_rm

R, h, E, nu = 5.0, 0.2, 3.44e9, 0.3
TAB = {"C11": 21606e6, "C22": 4153e6, "C33": 4153e6,
       "C44": 207650e6, "C55": 269680e6, "C66": 269680e6}


def D_iso():
    Q11 = E/(1-nu**2); Q12 = nu*Q11; Q66 = E/(2*(1+nu))
    Q = np.array([[Q11, Q12, 0.], [Q12, Q11, 0.], [0., 0., Q66]]); Z = np.zeros((3, 3))
    return np.block([[Q*h, Z], [Z, Q*h**3/12]])


def G_iso():
    return 5/6 * E/(2*(1+nu)) * h * np.eye(2)


def pipe_linear(n=80):
    th = np.array([2*np.pi*k/n for k in range(n)])
    nodes = np.column_stack([R*np.cos(th), R*np.sin(th)])
    elems = np.array([[k, (k+1) % n] for k in range(n)])
    k22 = -np.ones(n)/R
    return nodes, elems, ["iso"]*n, k22


def main():
    nodes, elems, lpe, k22 = pipe_linear(80)
    C6, EB = timoshenko_rm(nodes, elems, lpe, {"iso": D_iso()}, {"iso": G_iso()}, k22, p=1)
    nm = ["C11", "C22", "C33", "C44", "C55", "C66"]
    print("RM Timoshenko 6x6 (isotropic tube) vs Table 3.1:\n")
    print(f"  {'term':6s}{'RM V1':>14s}{'Table 3.1':>14s}{'% err':>9s}")
    for i, k in enumerate(nm):
        v = C6[i, i]
        print(f"  {k:6s}{v:14.4e}{TAB[k]:14.4e}{100*(v-TAB[k])/TAB[k]:9.2f}")
    print("\n  (C22=C33 are the transverse-shear terms -- the V1 payoff)")


if __name__ == "__main__":
    main()
