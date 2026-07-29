"""msg_rm_plate.py -- compatibility shim: the module moved to core OpenSG.

The MSG-RM plate law (Yu-2002 construction: classical ABD + least-squares transverse-shear
G over X = G^{-1} and Yu's 24 in-plane relaxed constants) now lives at
``opensg_jax.fe_jax.msg_rm_plate``.  This shim keeps the local
``from msg_rm_plate import rm_plate_msg`` imports of the xsec_paper scripts working.
"""
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from opensg_jax.fe_jax.msg_rm_plate import (_grad_ops, _lagrange_N, msgrm_strain_at_depth,
                                            rm_plate_msg)

__all__ = ["rm_plate_msg", "msgrm_strain_at_depth", "_grad_ops", "_lagrange_N"]
