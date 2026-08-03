"""Step 0 - generate the single-cell 1D-shell meshes (R/h = 1..10) into meshes/.

The single-cell tube is a plain circle of mean radius R = 0.0715 m with a single
[-45 deg] ply; the wall thickness h is swept so R/h = 1..10.  The two-cell webbed
meshes are shipped in meshes/ already (they need PreVABS to regenerate), so this
step only builds the single-cell circles, which are pure-numpy analytic.
"""
import os

from common import MESH, R_SINGLE, N_SINGLE, RH_LIST, ANI_MAT
import tube_lib  # noqa: F401  (ensures lib/ is importable)
from gen_meshes import gen_tube_yaml

for rh in RH_LIST:
    h = R_SINGLE / rh
    path = os.path.join(MESH, "shell_rh%02d.yaml" % rh)
    gen_tube_yaml(path, R_SINGLE, layup=[(-45.0, h)], mat=ANI_MAT, n=N_SINGLE, ccw=True)
    print("wrote  %-22s  R=%.4f  h=%.5f  R/h=%2d  N=%d" % (
        os.path.basename(path), R_SINGLE, h, rh, N_SINGLE))
print("single-cell meshes written to %s" % MESH)
