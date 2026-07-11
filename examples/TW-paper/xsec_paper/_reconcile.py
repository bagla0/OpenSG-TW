"""Reconcile the two solid references for IEA r=0.2 GA3 at full thickness:
mixed hex+tet solid at nr=4,6,8,10 vs the trusted PreVABS 2-D solid (C6_solid_r020.txt),
and the RM shell ring GA3 via ThinBlade(f=1)+emit vs the bundled shell_r020.yaml."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TAPER = os.path.abspath(os.path.join(HERE, "..", "..", "taper"))
TAPERJAX = os.path.abspath(os.path.join(HERE, "..", "..", "taper_jax"))
TWP = os.path.abspath(os.path.join(HERE, ".."))
for q in (MITC, REPO, TAPER, TAPERJAX, os.path.expanduser("~/OpenSG_io")):
    sys.path.insert(0, q)

from xsec_5v6_master import load_ring, load_solid, ring_6dof
from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block, emit_opensg_yaml
from opensg_io.hex_loft import hex_between_sections, solid_yaml_payload
from opensg_jax.fe_jax.solid_taper import (split_batches_to_tets, compute_timo_taper_solid_seg,
                                           _PERM3, _PERMF)
sys.path.insert(0, TAPERJAX)
ThinBlade = __import__("6_thin_02h_study").ThinBlade

blade = load_blade(WINDIO)
IB = os.path.join(TWP, "iea22_blade", "data")
So_prevabs = load_solid(os.path.join(IB, "C6_solid_r020.txt"))
print("PreVABS 2-D solid GA3 = %.4e" % So_prevabs[2, 2])

# bundled shell ring
Rb = load_ring(os.path.join(IB, "shell_r020.yaml"))
Cb = ring_6dof(Rb)
print("shell ring GA3 (bundled shell_r020) = %.4e  -> %%err vs PreVABS = %+.2f"
      % (Cb[2, 2], 100 * (Cb[2, 2] - So_prevabs[2, 2]) / So_prevabs[2, 2]))

# ThinBlade f=1 shell ring
cs = build_cross_section(ThinBlade(blade, 1.0), 0.2, mesh_size=0.02)
sp = os.path.join(HERE, "results", "_recon_shell.yaml"); emit_opensg_yaml(cs, sp)
Ct = ring_6dof(load_ring(sp))
print("shell ring GA3 (ThinBlade f=1 emit)   = %.4e  -> %%err vs PreVABS = %+.2f"
      % (Ct[2, 2], 100 * (Ct[2, 2] - So_prevabs[2, 2]) / So_prevabs[2, 2]))


def mixed_solid_ga3(nr):
    cs = build_cross_section(blade, 0.2, mesh_size=0.02)
    z1 = blade_span_z(blade, 0.2)
    res = hex_between_sections(cs, cs, z1, z1 + 2.0, nr=nr, nsp=6, nw=3, mesh_size=0.02)
    oris, hmats = solid_yaml_payload(res, cs, cs)
    web = np.array([t[0] == "web" for t in res["htag"]])
    names = sorted(set(hmats)); nix = {n: i for i, n in enumerate(names)}
    mp = []
    for n in names:
        e = _mat_block(blade, n)["elastic"]
        mp.append([e["E"][0], e["E"][1], e["E"][2], e["G"][0], e["G"][1], e["G"][2],
                   e["nu"][0], e["nu"][1], e["nu"][2]])
    seg = dict(nodes=np.asarray(res["nodes"])[:, _PERM3],
               batches={"hex8": (np.asarray(res["hexes"]),
                                 np.array([nix[m] for m in hmats], int),
                                 np.asarray(oris)[:, _PERMF])},
               mat_param=np.array(mp), nelem=len(res["hexes"]))
    seg = split_batches_to_tets(seg, mask=web)
    DL, _DR, _DS, _i = compute_timo_taper_solid_seg(seg, verbose=False)
    return 0.5 * (DL[2, 2] + DL[2, 2])


for nr in (4, 8, 12):
    g = mixed_solid_ga3(nr)
    print("mixed hex+tet solid GA3 (nr=%2d) = %.4e   (PreVABS %.4e ; ratio %.3f)"
          % (nr, g, So_prevabs[2, 2], g / So_prevabs[2, 2]), flush=True)
