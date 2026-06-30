"""JAX gradient-Kirchhoff (OpenSG-2.0) for the report's ANISOTROPIC [-45] tube --
gives the tension-torsion (extension-twist) COUPLING column for the tube table.

Report Case-2 (Opensg_MSG.pdf, Table 3.2 + Table 3.3): single [-45] ply,
R = 0.0715 m, h = 0.008682 m, E1=37, E2=E3=9 GPa, G12=G13=G23=4 GPa, nu=0.3,
CENTER reference, 92 curved elements.  Classical (EB 4x4) reference values (x10^6):
  C11(EA) 47.785 | C12(ext-twist) -0.93755 | C22(GJ) 0.14896 | C33=C44(EI) 0.10710
  (MSG-TW center); Yu et al. 2005 closed-form: 47.729 / -0.93607 / 0.14903 / 0.10728;
  VABS: 47.691 / -0.93541 / 0.14843 / 0.10690.

The 6x6 (order [EA,GA2,GA3,GJ,EI2,EI3]) maps to the report's classical order as
  C11=KF[0,0], C12(ext-twist)=KF[0,3], C22(GJ)=KF[3,3], C33=C44(EI)=KF[4,4].
"""
import os
import sys
import numpy as np

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
LIB = os.path.join(CC, "examples", "TW-paper", "lib")
sys.path.insert(0, LIB)

from gen_meshes import gen_tube_yaml  # noqa: E402
from tube_lib import homog            # noqa: E402

R_MEAN = 0.0715
H = 0.008682
ANI = {"E": [37.0e9, 9.0e9, 9.0e9], "G": [4.0e9, 4.0e9, 4.0e9], "nu": [0.3, 0.3, 0.3]}
LAYUP = [(-45.0, H)]
R_REF = R_MEAN
D_SHIFT = H / 2.0

# report Table 3.2 (x10^6), classical EB order [C11, C12(ext-twist), C22(GJ), C33=C44(EI)]
MSGTW = {"C11": 47.785e6, "C12": -0.93755e6, "C22": 0.14896e6, "C33": 0.10710e6}
YU05  = {"C11": 47.729e6, "C12": -0.93607e6, "C22": 0.14903e6, "C33": 0.10728e6}
VABS  = {"C11": 47.691e6, "C12": -0.93541e6, "C22": 0.14843e6, "C33": 0.10690e6}


def run(n_elem):
    datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inputs")
    os.makedirs(datadir, exist_ok=True)
    yaml_path = os.path.join(datadir, "aniso_tube_m45_n%d.yaml" % n_elem)
    gen_tube_yaml(yaml_path, R_REF, layup=LAYUP, mat=ANI, n=n_elem, ccw=True)
    _RM, KF = homog(yaml_path, R_ref=R_REF, d_shift=D_SHIFT, k22_mode="exact", e3="outward")
    return np.asarray(KF), yaml_path


def report(KF, n_elem):
    jx = {"C11": KF[0, 0], "C12": KF[0, 3], "C22": KF[3, 3], "C33": KF[4, 4]}
    print(f"\n=== JAX gradient-Kirchhoff ANISO [-45] tube, N={n_elem} ===")
    print(f"  full 6x6 order {['EA','GA2','GA3','GJ','EI2','EI3']}:")
    for i in range(6):
        print("   " + "".join(f"{KF[i, j]:13.4e}" for j in range(6)))
    print(f"\n  {'term':16s} {'JAX-KL':>12s} {'MSG-TW(FE)':>12s} {'Yu2005':>12s} {'VABS':>12s}")
    for k, nm in [("C11", "C11 (EA)"), ("C12", "C12 (ext-tw)"),
                  ("C22", "C22 (GJ)"), ("C33", "C33=C44 (EI)")]:
        print(f"  {nm:16s} {jx[k]:12.4e} {MSGTW[k]:12.4e} {YU05[k]:12.4e} {VABS[k]:12.4e}")
    print("\n  (values x10^6 for the paper table:)")
    for k, nm in [("C11", "C11"), ("C12", "C12"), ("C22", "C22"), ("C33", "C33=C44")]:
        print(f"   {nm:8s} JAX-KL = {jx[k]/1e6:10.5f}   MSG-TW = {MSGTW[k]/1e6:10.5f}"
              f"   Yu2005 = {YU05[k]/1e6:10.5f}   VABS = {VABS[k]/1e6:10.5f}")


if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1:]] or [92]
    for n in ns:
        KF, yp = run(n)
        report(KF, n)
        print(f"\n  yaml: {yp}")
