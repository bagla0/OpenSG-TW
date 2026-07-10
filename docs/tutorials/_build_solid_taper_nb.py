"""_build_solid_taper_nb.py -- build + EXECUTE docs/tutorials/solid_taper_jax.ipynb
(JAX solid taper: hex+tet hybrid tube, mixed boundaries, solid-vs-shell 4-way).

    python docs/tutorials/_build_solid_taper_nb.py     (run from repo root, opensg_2_0 env)
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook()
C = []

C.append(nbf.v4.new_markdown_cell(r"""# 11 · JAX solid taper — mixed hex+tet segment

The **3-D solid tapered segment** homogenized entirely in **JAX**
(`opensg_jax.fe_jax.solid_taper`): the FEniCS-OpenSG algorithm
(`compute_stiffness(Taper=True)` + `compute_timo_boun`) with **element-type
batches** — hex8 and tet4 volume elements are separate vmapped batches whose COO
triplets concatenate under one global DOF numbering, so a **mixed hex+tet segment
assembles natively into one system**.  The 2-D boundary cross-sections are
**extracted from the 3-D segment** (the JAX analogue of dolfinx `create_submesh`):
hex end-faces give quad4, tet end-faces give tri3 — a mixed segment therefore gets a
**mixed quad+tri boundary**, solved by the same batched 2-D SG (4-mode nullspace KKT).

Case: the tapered tube (thick wall $t/R=0.2$, single-ply $-45^\circ$, $a_R=0.7$,
`taper_study` mesh).  Three solid variants through the SAME solver — all-hex, a
**hybrid** with half the hoop split into tets ($x_2>0$, main-diagonal 6-tet split), and
all-tet — then the **RM shell** ring + tapered segment on the equivalent shell mesh.
Validation of this solver vs FEniCS on identical meshes: **0.008 %** (hex square),
**0.007 %** (hex m45), **0.63 %** (tet ellipse, diagonals ≤ 0.15 %)."""))

C.append(nbf.v4.new_code_cell(r"""import os, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "mitc_rm_segment"))
from opensg_jax.fe_jax.solid_taper import (read_solid_segment_yaml, split_batches_to_tets,
                                           compute_timo_taper_solid_seg)
ORDER = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]

def show(title, S, dt=None):
    print("\n%s%s" % (title, ("   [%.2f s]" % dt) if dt is not None else ""))
    for r in range(6):
        print("  " + "  ".join("% .4e" % S[r, c] for c in range(6)))

def pct(S, R):
    cut = np.abs(R).max() / 1e3                      # VABS max/1000 neglect rule
    E = np.full((6, 6), np.nan); m = np.abs(R) > cut
    E[m] = 100.0 * (S[m] - R[m]) / R[m]
    return E

def show_pct(title, E):
    print("\n%s  (%% err, max/1000 rule; . = negligible)" % title)
    for r in range(6):
        print("  " + "  ".join(("%8.3f" % E[r, c]) if np.isfinite(E[r, c]) else "     .  " for c in range(6)))
    print("  max |err| = %.3f %%" % np.nanmax(np.abs(E)))

TAG = "thick_m45_aR070"
MESH = os.path.join(ROOT, "mitc_rm_segment", "out", "taper_study", "meshes")
seg_hex = read_solid_segment_yaml(os.path.join(MESH, "solid_%s.yaml" % TAG))
print("tube %s : %d nodes, %d hex" % (TAG, len(seg_hex["nodes"]), seg_hex["nelem"]))"""))

C.append(nbf.v4.new_markdown_cell(r"""## Solid variants — HEX, HYBRID (hex+tet), TET

The hybrid splits every hex with centroid $x_2>0$ into 6 tets (main-diagonal scheme —
face diagonals match between neighbours, so the tet region is conforming; the hex|tet
interface is the standard node-tied transition).  **Both the segment and its extracted
boundaries are mixed-element** for the hybrid.  Each run prints the boundary L/R and
taper segment $6\times6$ with wall times."""))

C.append(nbf.v4.new_code_cell(r"""conn = seg_hex["batches"]["hex8"][0]
cent_x2 = seg_hex["nodes"][conn].mean(1)[:, 1]
variants = {"HEX": seg_hex,
            "HYBRID": split_batches_to_tets(seg_hex, mask=cent_x2 > 0.0),
            "TET": split_batches_to_tets(seg_hex)}
res = {}
for name, sg in variants.items():
    t0 = time.time()
    DL, DR, DS, info = compute_timo_taper_solid_seg(sg, verbose=False)
    dt = time.time() - t0
    bt = {k: len(v[0]) for k, v in sg["batches"].items()}
    print("\n======== solid %s : batches %s  dof=%d  total %.2f s ========" % (name, bt, info["dof"], dt))
    show("L boundary 6x6:", DL, info["t_boundary"] / 2)
    show("R boundary 6x6:", DR, info["t_boundary"] / 2)
    show("TAPER segment 6x6:", DS, info["t_segment"])
    res[name] = dict(L=DL, R=DR, seg=DS, dt=dt)"""))

C.append(nbf.v4.new_markdown_cell(r"""## RM shell — boundary rings + tapered shell segment

The equivalent shell mesh (`shell_thick_m45_aR070.yaml`, center reference) through the
general-RM taper machinery (`taper_study.shell_solve`, `mitc4_both`): ring $6\times6$
at L and R plus the tapered shell segment $6\times6$."""))

C.append(nbf.v4.new_code_cell(r"""t0 = time.time()
from taper_study import shell_solve
CL, S6, CR = shell_solve(TAG)
print("RM shell total %.2f s" % (time.time() - t0))
show("SHELL ring L 6x6:", CL)
show("SHELL ring R 6x6:", CR)
show("SHELL taper segment 6x6:", S6)
res["SHELL"] = dict(L=CL, R=CR, seg=S6)"""))

C.append(nbf.v4.new_markdown_cell(r"""## Comparisons

Hybrid and all-tet vs the all-hex reference (same solver, same geometry), then the RM
shell vs the solid for the boundary rings and the tapered segment.  The thick
($t/R=0.2$) $-45^\circ$ wall is the demanding regime for a shell — diagonals agree to a
few %, the large entries are the $-45^\circ$ coupling terms."""))

C.append(nbf.v4.new_code_cell(r"""show_pct("solid HYBRID vs solid HEX -- segment", pct(res["HYBRID"]["seg"], res["HEX"]["seg"]))
show_pct("solid TET    vs solid HEX -- segment", pct(res["TET"]["seg"], res["HEX"]["seg"]))
for part, nm in (("L", "L ring"), ("R", "R ring"), ("seg", "TAPER segment")):
    show_pct("RM SHELL vs solid HEX -- %s" % nm, pct(res["SHELL"][part], res["HEX"][part]))"""))

nb["cells"] = C
out = os.path.join(HERE, "solid_taper_jax.ipynb")
nbf.write(nb, out)
print("built", out)

import sys
rc = os.system("cd %s && %s -m nbconvert --to notebook --execute --inplace "
               "--ExecutePreprocessor.timeout=1800 solid_taper_jax.ipynb" % (HERE, sys.executable))
print("executed" if rc == 0 else "EXECUTE FAILED rc=%d" % rc, out)
