"""
Verify compute_ABD_matrix:
  (1) mesh-based reference shift (z_ref) vs the old parallel-axis transform
      shift_abd_reference -- at the mid-surface, for a symmetric and an
      UNSYMMETRIC ([0/90]) laminate (where B != 0 makes the reference matter);
  (2) prints the OML and centre ABD so they can be checked against the FEniCS
      OpenSG plate solver (fenics_abd.py, run in WSL).
Order [eps11, eps22, gam12 | k11, k22, k12].
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=False, linewidth=140)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "opensg_jax"))
from fe_jax import compute_ABD_matrix
from fe_jax.msg_materials import shift_abd_reference

ISO = {"iso": {"E": [70e9, 70e9, 70e9], "G": [26.923e9]*3, "nu": [0.3, 0.3, 0.3]}}
COMP = {"m": {"E": [37e9, 9e9, 9e9], "G": [4e9, 4e9, 4e9], "nu": [0.3, 0.3, 0.3]}}


def test(name, thick, angles, names, mat):
    h = float(sum(thick))
    abd_oml = np.asarray(compute_ABD_matrix(thick, angles, names, mat, z_ref=0.0)[0])
    abd_ctr_mesh = np.asarray(compute_ABD_matrix(thick, angles, names, mat, z_ref=h/2)[0])
    abd_ctr_pa = shift_abd_reference(abd_oml, h/2)
    scale = max(1.0, np.max(np.abs(abd_ctr_mesh)))
    rel = np.max(np.abs(abd_ctr_mesh - abd_ctr_pa)) / scale
    print(f"\n===== {name}  (h={h:g}) =====")
    print("B-block at OML (z_ref=0):\n", abd_oml[:3, 3:])
    print("B-block at CENTRE, mesh-shift (z_ref=h/2):\n", abd_ctr_mesh[:3, 3:])
    print("B-block at CENTRE, parallel-axis (shift_abd_reference):\n", abd_ctr_pa[:3, 3:])
    print(f"max |mesh - parallel-axis| / max|ABD|  =  {rel:.3e}")
    print("OML A11,B11,D11 = {:.5e}  {:.5e}  {:.5e}".format(
        abd_oml[0, 0], abd_oml[0, 3], abd_oml[3, 3]))
    print("ctr A11,B11,D11 = {:.5e}  {:.5e}  {:.5e}  (mesh-shift)".format(
        abd_ctr_mesh[0, 0], abd_ctr_mesh[0, 3], abd_ctr_mesh[3, 3]))


test("isotropic 1-ply", [0.01], [0.0], ["iso"], ISO)
test("[0/90] unsymmetric", [0.005, 0.005], [0.0, 90.0], ["m", "m"], COMP)
test("[45/-45] unsymmetric", [0.005, 0.005], [45.0, -45.0], ["m", "m"], COMP)
