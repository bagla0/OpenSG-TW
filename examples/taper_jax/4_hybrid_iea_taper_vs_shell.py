"""4_hybrid_iea_taper_vs_shell.py -- IEA r=0.2 -> r=0.3 TAPERED segment: SOLID MIXED
elements (hex skin + tet webs, JAX two-batch solver) vs the QUAD SHELL (ring-lofted).

SOLID: OpenSG_io hex loft (clean on this pair), web hexes split to tets (native hybrid
architecture; the L/R boundaries are MIXED quad4 skin + tri3 webs), homogenized by the
JAX solid taper (boundary L/R + segment 6x6, timed).

SHELL: OpenSG_io lofts the QUAD shell taper FROM THE BOUNDARY RINGS (the 1-D contours
at r1/r2 -- identical ring topology at both stations, OML reference, refinable via
NSP/MESH; quad-only for now, tri/mixed surface elements deferred).  Solved by the
OpenSG-TW MITC-RM machinery (run_indep.shell_solve_lagrange_sparse), which returns the
two end-RING 6x6 plus the tapered shell segment 6x6.

Prints all six 6x6 with wall times, then the solid-vs-shell %err tables for L ring,
R ring, and the tapered segment.  Writes iea_r020_030_hybrid_vs_shell.dat.

    python 4_hybrid_iea_taper_vs_shell.py [r1 r2]        (default 0.2 0.3)
"""
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "taper"))
sys.path.insert(0, os.path.join(ROOT, "mitc_rm_segment"))
sys.path.insert(0, os.path.expanduser("~/OpenSG_io"))

import yaml
from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block
from opensg_io.hex_loft import (hex_between_sections, shell_between_sections,
                                shell_yaml_payload, shell_boundary_payload,
                                assert_shell_conforming, solid_yaml_payload)
from opensg_io.conformity import min_scaled_jacobian
from opensg_jax.fe_jax.solid_taper import (split_batches_to_tets, compute_timo_taper_solid_seg,
                                           _PERM3, _PERMF)
import run_indep

ORDER = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
R1 = float(sys.argv[1]) if len(sys.argv) > 2 else 0.2
R2 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
NR, NSP, NW, MESH = 4, 12, 3, 0.02
OUT = os.path.join(HERE, "out_iea_taper")
os.makedirs(OUT, exist_ok=True)


def show(title, S, dt=None):
    print("\n%s%s" % (title, ("   [%.1f s]" % dt) if dt is not None else ""))
    for r in range(6):
        print("  " + "  ".join("% .5e" % S[r, c] for c in range(6)))
    print("  diagonal: " + "  ".join("%s=%.4e" % (ORDER[i], S[i, i]) for i in range(6)))


def pct(S, Rf):
    cut = np.abs(Rf).max() / 1e3
    E = np.full((6, 6), np.nan)
    m = np.abs(Rf) > cut
    E[m] = 100.0 * (S[m] - Rf[m]) / Rf[m]
    return E


def show_pct(title, E):
    print("\n%s  (%% err, max/1000 rule; . = negligible)" % title)
    for r in range(6):
        print("  " + "  ".join(("%8.3f" % E[r, c]) if np.isfinite(E[r, c]) else "     .  "
                               for c in range(6)))
    print("  max |err| = %.3f %%" % np.nanmax(np.abs(E)))


# ================================================================= SOLID mixed (hex+tet)
print("=== SOLID: hex loft r=%.2f->%.2f + web-split -> HYBRID ===" % (R1, R2), flush=True)
blade = load_blade(WINDIO)
cs1 = build_cross_section(blade, R1, mesh_size=MESH)
cs2 = build_cross_section(blade, R2, mesh_size=MESH)
z1, z2 = blade_span_z(blade, R1), blade_span_z(blade, R2)
res = hex_between_sections(cs1, cs2, z1, z2, nr=NR, nsp=NSP, nw=NW, mesh_size=MESH)
nodes, hexes, htag = res["nodes"], np.asarray(res["hexes"]), res["htag"]
msj, ninv = min_scaled_jacobian(nodes, hexes)
assert ninv == 0, "hex loft inverted on this pair (%d)" % ninv
oris, hmats = solid_yaml_payload(res, cs1, cs2)
web_mask = np.array([t[0] == "web" for t in htag])

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
seg_hyb = split_batches_to_tets(seg, mask=web_mask)
bt = {k: len(v[0]) for k, v in seg_hyb["batches"].items()}
print("solid hybrid: %d nodes; batches %s (min SJ %.3f)" % (len(nodes), bt, msj), flush=True)

t0 = time.time()
DLs, DRs, DSs, info = compute_timo_taper_solid_seg(seg_hyb, verbose=True)
t_solid = time.time() - t0
show("SOLID hybrid L-boundary 6x6 (mixed quad+tri):", DLs, info["t_boundary"] / 2)
show("SOLID hybrid R-boundary 6x6:", DRs, info["t_boundary"] / 2)
show("SOLID hybrid TAPER segment 6x6:", DSs, info["t_segment"])

# ============================================================ SHELL quad from boundary rings
print("\n=== SHELL: quad taper lofted from the boundary rings (OML ref) ===", flush=True)
shell = shell_between_sections(res, cs1, cs2, reference="OML")
assert_shell_conforming(shell, len(cs1["webs"]), NSP)
seg_yaml = os.path.join(OUT, "shell_segment.yaml")
yaml.safe_dump(shell_yaml_payload(shell, blade, _mat_block), open(seg_yaml, "w"),
               default_flow_style=None, sort_keys=False)
for side, si in (("L", 0), ("R", 1)):
    yaml.safe_dump(shell_boundary_payload(res, shell, cs1, cs2, si, blade, _mat_block),
                   open(os.path.join(OUT, "shell_boundary_%s.yaml" % side), "w"),
                   default_flow_style=None, sort_keys=False)
print("shell quad segment: %d nodes / %d quads (rings at both stations, quad-only)"
      % (len(shell["nodes"]), len(shell["quads"])), flush=True)

work = os.path.join(OUT, "_shellrun")
os.makedirs(work, exist_ok=True)
shutil.copy(seg_yaml, os.path.join(work, "shell_seg.yaml"))
t0 = time.time()
r = run_indep.shell_solve_lagrange_sparse("seg", work, work, shear="full", return_full=True)
t_shell = time.time() - t0
sym = lambda M: 0.5 * (np.asarray(M) + np.asarray(M).T)
CL, CR, S6 = sym(r["C6L"]), sym(r["C6R"]), sym(r["S6"])
show("SHELL ring L 6x6:", CL, r["t_rings"] / 2)
show("SHELL ring R 6x6:", CR, r["t_rings"] / 2)
show("SHELL TAPER segment 6x6:", S6, r["t_seg"])

# ================================================================= comparison tables
print("\n########## SOLID MIXED (hex skin + tet webs) vs QUAD SHELL, r=%.2f->%.2f ##########"
      % (R1, R2))
show_pct("SHELL vs SOLID -- L boundary/ring", pct(CL, DLs))
show_pct("SHELL vs SOLID -- R boundary/ring", pct(CR, DRs))
show_pct("SHELL vs SOLID -- TAPER segment", pct(S6, 0.5 * (DSs + DSs.T)))

dat = os.path.join(HERE, "iea_r020_030_hybrid_vs_shell.dat")
with open(dat, "w") as f:
    f.write("# IEA r=%.2f->%.2f taper: SOLID MIXED (hex skin + tet webs, JAX) vs QUAD SHELL"
            " (ring-lofted, MITC-RM)\n" % (R1, R2))
    f.write("# solid batches %s ; solid dof %d ; solid %.1f s ; shell %.1f s\n"
            % (bt, info["dof"], t_solid, t_shell))
    f.write("# diagonals [EA GA2 GA3 GJ EI2 EI3]\n")
    for nm, M in (("solid_L", DLs), ("solid_R", DRs), ("solid_seg", DSs),
                  ("shell_L", CL), ("shell_R", CR), ("shell_seg", S6)):
        f.write("%-10s " % nm + " ".join("%.6e" % M[i, i] for i in range(6)) + "\n")
    f.write("# max %%err shell-vs-solid: L %.3f  R %.3f  seg %.3f\n"
            % (np.nanmax(np.abs(pct(CL, DLs))), np.nanmax(np.abs(pct(CR, DRs))),
               np.nanmax(np.abs(pct(S6, 0.5 * (DSs + DSs.T))))))
print("\nwrote", dat)
