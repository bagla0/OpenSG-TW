"""
RM thin-walled beam — step 1: closed-form classical (EB) validation.

Uses the RM-derived SOLVED shell strains for the isotropic symmetric laminate at
the CENTER reference (Opensg_MSG eq 4.28) and integrates the classical shell
strain energy over the circular-tube contour:

    Gamma_D(s) = B(s) @ [gamma11, kappa1, kappa2, kappa3]
    C_EB       = INT_contour  B(s)^T  D6  B(s)  ds

and compares the diagonal to Table 3.1 (isotropic circular tube, center ref).

Captured in closed form: EA (C11) and bending (C55=C66).  GJ (C44) of a CLOSED
tube is the Bredt shear flow, which lives in the boundary (closed-loop) terms of
eq 4.26 and needs the full FE fluctuation solve -- so C44 here is deliberately
wrong and motivates rm/msg_rm.py (step 2).
"""
import numpy as np

# isotropic circular tube (Opensg_MSG Table 3.1, center reference)
R, h, E, nu = 5.0, 0.2, 3.44e9, 0.3
TABLE = {"C11 (EA)": 21606e6, "C44 (GJ)": 207650e6,
         "C55 (bend2)": 269680e6, "C66 (bend3)": 269680e6}


def D6_isotropic(E, nu, h):
    Q11 = E / (1 - nu**2); Q12 = nu * Q11; Q66 = E / (2 * (1 + nu))
    Q = np.array([[Q11, Q12, 0.0], [Q12, Q11, 0.0], [0.0, 0.0, Q66]])
    A = Q * h; Db = Q * h**3 / 12.0; Z = np.zeros((3, 3))
    return np.block([[A, Z], [Z, Db]])               # symmetric laminate, B=0


def B_eq428(x2, x3, t2, t3, nu):
    """Gamma_D = [e11,e22,2e12,k11,k22,k12+k21] = B @ [g11,k1,k2,k3]  (eq 4.28).

    x = position, t = unit tangent (dx/ds).  Center ref, isotropic."""
    e11 = np.array([1.0, 0.0, x3, -x2])              # g11 + x3 k2 - x2 k3
    B = np.zeros((6, 4))
    B[0] = e11                                       # e11
    B[1] = -nu * e11                                 # e22 = -nu e11
    B[2] = 0.0                                       # 2e12 = 0
    B[3] = np.array([0.0, 0.0, t2, t3])              # k11 = t2 k2 + t3 k3
    B[4] = -nu * B[3]                                # k22 = -nu k11
    B[5] = np.array([0.0, -2.0, 0.0, 0.0])           # k12+k21 = -2 k1
    return B


def main():
    D6 = D6_isotropic(E, nu, h)
    th, w = np.polynomial.legendre.leggauss(400)
    th = np.pi * (th + 1.0); w = np.pi * w            # map [-1,1] -> [0,2pi]
    C = np.zeros((4, 4))
    for t, wq in zip(th, w):
        x2, x3 = R * np.cos(t), R * np.sin(t)
        t2, t3 = -np.sin(t), np.cos(t)                # unit tangent
        B = B_eq428(x2, x3, t2, t3, nu)
        C += B.T @ D6 @ B * (R * wq)                  # ds = R dtheta

    names = ["C11 (EA)", "C44 (GJ)", "C55 (bend2)", "C66 (bend3)"]
    print("RM closed-form classical (eq 4.28), isotropic tube, center ref:\n")
    print(f"  {'term':14s}{'closed-form':>15s}{'Table 3.1':>15s}{'% err':>9s}  note")
    for i, nm in enumerate(names):
        val = C[i, i]; ref = TABLE[nm]
        note = ("EA/EI: captured" if "GJ" not in nm
                else "needs full solve (closed-section Bredt shear flow)")
        print(f"  {nm:14s}{val:15.4e}{ref:15.4e}{100*(val-ref)/ref:9.2f}  {note}")
    print("\n=> EA and bending match Table 3.1 in closed form; GJ requires the RM "
          "FE solve (step 2, rm/msg_rm.py).")


if __name__ == "__main__":
    main()
