"""Validate rm/msg_rm.py on the isotropic circular tube (Opensg_MSG Table 3.1,
center reference).  EB 4x4 -> EA (C11), GJ (C44), EI (C55=C66).  Transverse-shear
C22=C33 needs the V1/Timoshenko step (not yet)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from msg_rm import assemble_rm, solve_eb

R, h, E, nu = 5.0, 0.2, 3.44e9, 0.3
TABLE = {"EA (C11)": 21606e6, "GJ (C44)": 207650e6, "EI (C55)": 269680e6}


def D_iso(E, nu, h):
    Q11 = E/(1-nu**2); Q12 = nu*Q11; Q66 = E/(2*(1+nu))
    Q = np.array([[Q11, Q12, 0.], [Q12, Q11, 0.], [0., 0., Q66]])
    Z = np.zeros((3, 3))
    return np.block([[Q*h, Z], [Z, Q*h**3/12]])


def G_iso(E, nu, h, kcorr=5/6):
    G = E/(2*(1+nu))
    return kcorr * G * h * np.eye(2)


def circle_mesh(R, n):
    """closed quadratic (3-node) contour, center-ref nodes at radius R."""
    nodes = np.zeros((2*n, 2))
    for k in range(2*n):
        th = np.pi * k / n
        nodes[k] = [R*np.cos(th), R*np.sin(th)]
    elems = np.array([[2*e, 2*e+1, (2*e+2) % (2*n)] for e in range(n)])
    k22 = -np.ones(n) / R                      # contour curvature -1/R
    return nodes, elems, k22


def run(n=60, reduced=True):
    nodes, elems, k22 = circle_mesh(R, n)
    D = D_iso(E, nu, h); Gs = G_iso(E, nu, h)
    Kqq, Kqe, Kee = assemble_rm(nodes, elems, 5*len(nodes), D, Gs, k22, p=2, reduced=reduced)
    C = solve_eb(Kqq, Kqe, Kee, nodes)
    return C


def main():
    global h
    print("Isotropic tube, RM (C0), center ref vs Table 3.1 (h=0.2, R/h=25)\n")
    for tag, red in [("reduced int (locking fix)", True), ("full int", False)]:
        C = run(60, red)
        vals = {"EA (C11)": C[0, 0], "GJ (C44)": C[1, 1], "EI (C55)": C[2, 2]}
        print(f"  --- {tag} ---")
        for k in TABLE:
            print(f"    {k:10s} {vals[k]:13.4e}  ref {TABLE[k]:13.4e}  "
                  f"{100*(vals[k]-TABLE[k])/TABLE[k]:+7.2f}%")
        print(f"    EI2-EI3 sym: {abs(C[2,2]-C[3,3])/C[2,2]:.1e}\n")

    # --- locking check: thin wall, coarse mesh ---
    print("Locking check (R/h=500, n=20): full vs reduced integration of the G-energy")
    h_save = h; h = R/500.0
    EI_thin = E*h*np.pi*R**3 + E*h*(np.pi*R)*h**2/12
    Ef, Er = run(20, False)[2, 2], run(20, True)[2, 2]
    print(f"    full int   EI={Ef:13.4e}  err {100*(Ef-EI_thin)/EI_thin:+6.2f}%")
    print(f"    reduced    EI={Er:13.4e}  err {100*(Er-EI_thin)/EI_thin:+6.2f}%")
    print(f"    => full==reduced ({abs(Ef-Er)/Er:.1e}): these beam sections are")
    print("       MEMBRANE-dominated (EI from eps11=x3*k2), so the plate-bending /")
    print("       transverse-shear terms (where locking lives) are a minor part and")
    print("       do not lock.  Reduced integration is kept as the safeguard for")
    print("       any bending-dominated region (e.g. very thick webs/caps).")
    h = h_save


if __name__ == "__main__":
    main()
