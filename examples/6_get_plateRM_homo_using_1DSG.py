"""Example 6 - MSG-RM plate homogenization: the 8x8 ABDG wall law from a 1-D shell YAML.

Reads the BAR-URC station-15 1-D shell SG YAML and, for every wall laminate (elementSet), runs the
through-thickness MSG-RM construction (Yu, Hodges & Volovoi, Computers & Structures 81, 2003,
Sec. 4): the classical ABD 6x6 from the zeroth-order warping (Eq. 40), and the transverse-shear
2x2 G from the least-squares RM projection of the second-order gradient energy (Eqs. 55-60;
27 unknowns = G^-1 + Yu's 24 in-plane relaxed constants).  Prints the full 8x8 RM plate law
ABDG = [[A,B,0],[B,D,0],[0,0,G]] per layup at the CENTER (mid-surface) reference, the residual
Ustar_rel (fraction of the second-order energy the RM form could not absorb), and G_msg vs the
Whitney complementary-energy shear stiffness.

Run (Windows, env on PATH):
    python examples/6_get_plateRM_homo_using_1DSG.py
"""
import os
import sys
import time

import numpy as np

CC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("", "opensg_jax"):
    sys.path.insert(0, os.path.join(CC, p))
np.set_printoptions(precision=4, linewidth=150, suppress=False)

from opensg_jax.fe_jax.msg_mesh import load_yaml
from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg_batch
from opensg_jax.fe_jax.msg_transverse_shear import plate_8x8, transverse_shear_stiffness

SHELL = os.path.join(CC, "examples", "data", "1d_yaml", "st15_shell.yaml")


def main():
    _, _, mdb, layup_db, elem_to_layup = load_yaml(SHELL)
    n_elem = {ln: sum(1 for v in elem_to_layup.values() if v == ln) for ln in layup_db}
    print("1-D SG: %s  (%d wall laminates)" % (os.path.relpath(SHELL, CC), len(layup_db)))
    # all laminates in ONE vmapped pass (grouped into ply-count buckets)
    names = list(layup_db)
    t0 = time.perf_counter()
    res = rm_plate_msg_batch([layup_db[ln] for ln in names], mdb, fraction=0.5)  # 0=OML, 1=IML
    print("rm_plate_msg_batch: %d laminates in %.3f s (incl. one-time jit per bucket)"
          % (len(res), time.perf_counter() - t0))
    for ln, r in zip(names, res):
        lay = layup_db[ln]
        thk = [float(t) for t in lay["thick"]]
        ang = [float(a) for a in lay["angles"]]
        mats = [str(m) for m in lay["mat_names"]]
        h = float(sum(thk))
        Gw = transverse_shear_stiffness(thk, ang, mats, mdb)[0]
        G = r["G_msg"] if r["G_msg"] is not None else Gw
        P8 = plate_8x8(r["A6"], G)
        print("\n== %s : %d plies, h = %.4f m, %d elements ==" % (ln, len(thk), h, n_elem[ln]))
        print("   plies:", ", ".join("%s(%.2fmm/%g)" % (m, 1e3 * t, a)
                                     for m, t, a in zip(mats, thk, ang)))
        print("   RM 8x8 ABDG [[A,B,0],[B,D,0],[0,0,G]]"
              " (rows 1-6: e11,e22,g12,k11,k22,k12; rows 7-8: 2g13,2g23):")
        print(P8)
        print("   G_msg diag = [%.4e %.4e]   Whitney = [%.4e %.4e]   Ustar_rel = %.2e%s"
              % (G[0, 0], G[1, 1], Gw[0, 0], Gw[1, 1], r["Ustar_rel"],
                 "" if r["G_msg"] is not None else "   (X not SPD -> Whitney fallback)"))


if __name__ == "__main__":
    main()
