"""3_prismatic_iea_hybrid_check.py -- PRISMATIC MSG identity check on a MIXED hex+tet
segment of the real IEA-22 r=0.2 cross-section: hex SKIN + tet WEBS (the native hybrid
architecture), boundary therefore MIXED quad4(skin) + tri3(webs).

For a PRISMATIC segment the homogenized segment 6x6 must EQUAL the boundary 6x6 (the
classic MSG consistency identity) -- this validates the mixed-element segment + mixed
boundary path end-to-end on blade geometry.

Builds the prismatic segment in-memory with OpenSG_io (build_cross_section at r=0.1967 +
hex_between_sections with cs1=cs2), tags web hexes via the generator's own ftag, splits
EXACTLY the web hexes into tets (split_batches_to_tets), runs the JAX solid taper, and
compares segment vs L/R boundary 6x6.

    python 3_prismatic_iea_hybrid_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.expanduser("~/OpenSG_io"))

from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import hex_between_sections, solid_yaml_payload
from opensg_jax.fe_jax.solid_taper import (split_batches_to_tets, compute_timo_taper_solid_seg,
                                           _PERM3, _PERMF)

ORDER = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
R = 0.1967
LSEG = 2.0
WINDIO = os.path.expanduser("~/OpenSG_io/examples/data/IEA-22-280-RWT.yaml")


def show(title, S):
    print("\n%s" % title)
    for r in range(6):
        print("  " + "  ".join("% .5e" % S[r, c] for c in range(6)))
    print("  diagonal: " + "  ".join("%s=%.4e" % (ORDER[i], S[i, i]) for i in range(6)))


def pct(S, Rf):
    cut = np.abs(Rf).max() / 1e3
    E = np.full((6, 6), np.nan)
    m = np.abs(Rf) > cut
    E[m] = 100.0 * (S[m] - Rf[m]) / Rf[m]
    return E


# ---------------------------------------------------------------- build prismatic segment
print("building PRISMATIC IEA r=%.4f segment (hex loft, cs1=cs2) ..." % R, flush=True)
blade = load_blade(WINDIO)
cs = build_cross_section(blade, R, mesh_size=0.02)
z1 = R * 137.0
res = hex_between_sections(cs, cs, z1, z1 + LSEG, nr=4, nsp=6, nw=3, mesh_size=0.02)
nodes, hexes, htag = res["nodes"], np.asarray(res["hexes"]), res["htag"]
oris, hmats = solid_yaml_payload(res, cs, cs)
web_mask = np.array([t[0] == "web" for t in htag])
print("prismatic segment: %d nodes, %d hexes (%d web -> tets, %d skin stay hex)"
      % (len(nodes), len(hexes), web_mask.sum(), (~web_mask).sum()), flush=True)

# ---------------------------------------------------------------- JAX seg dict (beam-first)
mat_names = sorted(set(hmats))
name_ix = {n: i for i, n in enumerate(mat_names)}
mat_param = []
for n in mat_names:
    e = _mat_block(blade, n)["elastic"]
    mat_param.append([e["E"][0], e["E"][1], e["E"][2], e["G"][0], e["G"][1], e["G"][2],
                      e["nu"][0], e["nu"][1], e["nu"][2]])
seg = dict(nodes=np.asarray(nodes)[:, _PERM3],
           batches={"hex8": (hexes, np.array([name_ix[m] for m in hmats], int),
                             np.asarray(oris)[:, _PERMF])},
           mat_param=np.array(mat_param), nelem=len(hexes))

seg_hyb = split_batches_to_tets(seg, mask=web_mask)          # hex SKIN + tet WEBS
bt = {k: len(v[0]) for k, v in seg_hyb["batches"].items()}
print("hybrid batches:", bt, flush=True)

# ---------------------------------------------------------------- solve + prismatic identity
DL, DR, DS, info = compute_timo_taper_solid_seg(seg_hyb, verbose=True)
show("HYBRID L-boundary 6x6 (mixed quad skin + tri webs):", DL)
show("HYBRID R-boundary 6x6:", DR)
show("HYBRID PRISMATIC SEGMENT 6x6:", DS)

print("\n########## PRISMATIC MSG IDENTITY (segment must equal boundary) ##########")
EL = pct(0.5 * (DS + DS.T), DL)
ER = pct(0.5 * (DS + DS.T), DR)
print("\nsegment vs L boundary (%err, max/1000 rule):")
for r in range(6):
    print("  " + "  ".join(("%8.4f" % EL[r, c]) if np.isfinite(EL[r, c]) else "     .  "
                           for c in range(6)))
print("  max |err| segment-vs-L = %.4f %%" % np.nanmax(np.abs(EL)))
print("  max |err| segment-vs-R = %.4f %%" % np.nanmax(np.abs(ER)))
print("  max |err| L-vs-R       = %.4f %%" % np.nanmax(np.abs(pct(DL, DR))))
