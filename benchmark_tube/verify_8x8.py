"""Verify compute_ABD_matrix(shear_refined=...) -> 6x6 default / 8x8 RM, and that
the 8x8 G block matches transverse_shear_stiffness; check default is unchanged."""
import os, sys
import numpy as np
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
from fe_jax import compute_ABD_matrix, transverse_shear_stiffness

ISO = {"E": [70e9]*3, "G": [26.923e9]*3, "nu": [0.3]*3}
ANI = {"E": [37e9, 9e9, 9e9], "G": [4e9, 5e9, 3e9], "nu": [0.3]*3}   # G13!=G23 to exercise coupling
mats = {"iso": ISO, "ani": ANI}

for name, layup in [("iso 1-ply", ([0.1], [0.0], ["iso"])),
                    ("[45/-45] ani (G13!=G23)", ([0.05, 0.05], [45.0, -45.0], ["ani", "ani"]))]:
    th, ang, nm = layup
    abd6, _ = compute_ABD_matrix(th, ang, nm, mats)                    # default
    p8, _ = compute_ABD_matrix(th, ang, nm, mats, shear_refined=True)  # RM 8x8
    Gref = transverse_shear_stiffness(th, ang, nm, mats)[0]            # MSG default
    print(f"\n=== {name} ===")
    print(f"  default shape  = {abd6.shape}   (expect (6, 6))")
    print(f"  refined shape  = {p8.shape}   (expect (8, 8))")
    print(f"  8x8 top-left == ABD ?  {np.allclose(p8[:6, :6], abd6)}")
    print(f"  8x8 off-blocks zero ?  {np.allclose(p8[:6, 6:], 0) and np.allclose(p8[6:, :6], 0)}")
    print(f"  8x8 G block == transverse_shear_stiffness ?  {np.allclose(p8[6:, 6:], Gref)}")
    print(f"  G block =\n{np.array2string(p8[6:, 6:], precision=4)}")
