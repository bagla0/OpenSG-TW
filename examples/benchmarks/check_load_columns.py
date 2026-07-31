"""Validate the new LOAD COLUMNS: sigma33 DIRECT from the MSG recovery (Yu's route)
vs exact 3-D, Garg caseA/C.  Also the face-traction check that fixes the sign."""
import os, sys
import numpy as np

ROOT = os.path.expanduser("~/OpenSG-TW-claude")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "garg"))
import jax
jax.config.update("jax_enable_x64", True)

from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth
from pagano_exact import pagano_profiles
from statics_fsdt import statics_resultants
from garg_layups import MATERIAL_DB, LAYUPS, H

q0 = 1.0e4; a = 1.0; p = np.pi / a

for case in ("caseA", "caseC"):
    for S in (4, 10, 100):
        h = a / S
        lay = LAYUPS[case]
        fr = [t / H for t in lay["thick"]]
        thk = [f * h for f in fr]; ang = lay["angles"]; mats = lay["mat_names"]
        zc, sig, _ = pagano_profiles(thk, ang, mats, MATERIAL_DB, a=a, q0=q0)
        r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
        S6 = np.linalg.inv(np.asarray(r["A6"]))
        Q1, M11 = statics_resultants(q0, a, a / 2)      # x = a/2: Q1 = 0, M11 = q0/p^2
        # plate state at x = a/2 (sin peak): E from M; gradients of E vanish (Q=0),
        # BUT the load ladder is active: q(a/2) = q0, q,1 = 0, q,11 = -p^2 q0
        FF6 = np.array([0, 0, 0, M11, 0, 0.0])
        E6 = S6 @ FF6
        s33_d = np.empty_like(zc); s13_x0 = np.empty_like(zc)
        dE1_end = S6 @ np.array([0, 0, 0, q0 / p, 0, 0.0])
        dE11 = -p * p * E6                  # sin family: E,11(a/2) = -p^2 E(a/2)
        for i, z in enumerate(zc):
            # sigma33 station x = a/2 -- DIRECT recovery: strain ladder (V2 blocks,
            # driven by E,11) PLUS load ladder (V1L q + V2L11 q,11)
            s33_d[i] = msgrm_strain_at_depth(r, z, E6, None, None, dE11=dE11,
                                             q=q0, dq11=-p * p * q0)[1][2]
            # sigma13 station x = 0 -- q(0)=0, q,1(0) = p q0
            s13_x0[i] = msgrm_strain_at_depth(r, z, np.zeros(6), dE1_end, None,
                                              dq1=p * q0)[1][4]

        def relerr(m, e):
            return 100 * np.linalg.norm(m - e) / np.linalg.norm(e)

        print("%s S=%-4g  s33 DIRECT: err %7.3f%%  faces (bot, top)/q0 = (%+.4f, %+.4f)"
              "   s13(+load): err %7.3f%%"
              % (case, S, relerr(s33_d, sig[:, 2]), s33_d[0] / q0, s33_d[-1] / q0,
                 relerr(s13_x0, sig[:, 4])))
