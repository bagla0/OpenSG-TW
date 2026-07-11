"""thick_sweep_r020.py -- THICKNESS sweep at IEA r=0.2: scale EVERY lamina thickness by
f in {0.2,0.4,0.6,0.8,1.0} and track the cross-section %err per Timo term, for both RM
cross-section formulations (6-DOF Lagrange g23, 5-DOF MITC g23) vs the 2-D solid.

Shows the RM cross-section stays accurate across the thin->thick wall range (t/h from
~0.2h to 1.0h of the design laminate).

Per f:
  * shell ring  = OpenSG_io build_cross_section(ThinBlade(blade,f), 0.2) -> emit -> ring
  * 2-D solid   = PRISMATIC mixed hex+tet cross-section at r=0.2, factor f, homogenized
                  by the JAX solid taper (boundary DL == the 2-D solid cross-section;
                  prismatic identity DL=DR=segment verified this session).

    python thick_sweep_r020.py
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TAPERJAX = os.path.abspath(os.path.join(HERE, "..", "..", "taper_jax"))
TAPER = os.path.abspath(os.path.join(HERE, "..", "..", "taper"))
for q in (MITC, REPO, TAPERJAX, TAPER, os.path.expanduser("~/OpenSG_io")):
    sys.path.insert(0, q)

from xsec_5v6_master import load_ring, ring_6dof, ring_5dof, LBL
from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block, emit_opensg_yaml
from opensg_io.hex_loft import hex_between_sections, solid_yaml_payload
from opensg_jax.fe_jax.solid_taper import (split_batches_to_tets, compute_timo_taper_solid_seg,
                                           _PERM3, _PERMF)

# ThinBlade wrapper (uniform ply scaling) -- same as the 0.2h study
sys.path.insert(0, TAPERJAX)
ThinBlade = __import__("6_thin_02h_study").ThinBlade

R = 0.2
FS = [0.2, 0.4, 0.6, 0.8, 1.0]
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
SCR = os.path.join(OUT, "_thick_scratch"); os.makedirs(SCR, exist_ok=True)
blade0 = load_blade(WINDIO)


def solid_xsec(blade, f):
    """2-D solid cross-section 6x6 at r=0.2, ply factor f: prismatic mixed hex+tet."""
    cs = build_cross_section(blade, R, mesh_size=0.02)
    z1 = blade_span_z(blade, R)
    res = hex_between_sections(cs, cs, z1, z1 + 2.0, nr=4, nsp=6, nw=3, mesh_size=0.02)
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
    DL, _DR, _DS, _info = compute_timo_taper_solid_seg(seg, verbose=False)
    return 0.5 * (DL + DL.T)


rows = []
for f in FS:
    tb = ThinBlade(blade0, f)
    # shell ring
    cs = build_cross_section(tb, R, mesh_size=0.02)
    sp = os.path.join(SCR, "shell_r020_f%02d.yaml" % round(f * 10))
    emit_opensg_yaml(cs, sp)
    ring = load_ring(sp)
    C6 = ring_6dof(ring); C5 = ring_5dof(ring)
    So = solid_xsec(tb, f)
    e6 = [100.0 * (C6[i, i] - So[i, i]) / So[i, i] for i in range(6)]
    e5 = [100.0 * (C5[i, i] - So[i, i]) / So[i, i] for i in range(6)]
    rows.append((f, e6, e5, So.copy(), C6.copy(), C5.copy()))
    print("f=%.1fh  6-DOF %s | 5-DOF %s"
          % (f, " ".join("%+5.2f" % v for v in e6), " ".join("%+5.2f" % v for v in e5)), flush=True)

np.savez(os.path.join(OUT, "thick_sweep_r020.npz"),
         f=np.array([r[0] for r in rows]),
         err6=np.array([r[1] for r in rows]), err5=np.array([r[2] for r in rows]),
         labels=LBL)
print("\nheader: f  " + "  ".join(LBL))
print("wrote", os.path.join(OUT, "thick_sweep_r020.npz"))
