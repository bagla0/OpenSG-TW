"""timing_r020.py -- IEA r=0.2 cross-section: wall-clock COST of the three homogenizers
across mesh-refinement (convergence) levels:
  * RM 6-DOF shell ring   (opensg_jax ring_indep)         -- 1-D contour
  * JAX  2-D solid         (opensg_jax solid_taper)        -- 3-D prismatic hex boundary
  * FEniCS 2-D solid       (opensg.core.solid.compute_stiffness, Taper=False)  -- same mesh
Same geometry per level (a prismatic hex cross-section at r=0.2); refinement = in-plane
mesh size + through-thickness element count nr.  Prints DOF, wall time, and a diagonal
Timo term to confirm convergence.

    python timing_r020.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TAPER = os.path.abspath(os.path.join(HERE, "..", "..", "taper"))
for q in (MITC, REPO, TAPER, os.path.expanduser("~/OpenSG_io")):
    sys.path.insert(0, q)

from xsec_5v6_master import load_ring, ring_6dof
from taper_common import WINDIO, blade_span_z
from opensg_io.converter import load_blade, build_cross_section, _mat_block, emit_opensg_yaml
from opensg_io.hex_loft import hex_between_sections, solid_yaml_payload
from opensg_jax.fe_jax.solid_taper import (split_batches_to_tets, extract_boundary_submesh,
                                           solve_boundary, _PERM3, _PERMF, write_solid_segment_yaml)


def _mp_row(blade, n):
    e = _mat_block(blade, n)["elastic"]
    return [e["E"][0], e["E"][1], e["E"][2], e["G"][0], e["G"][1], e["G"][2],
            e["nu"][0], e["nu"][1], e["nu"][2]]

OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
SCR = os.path.join(OUT, "_timing_scr"); os.makedirs(SCR, exist_ok=True)
R = 0.2
LEVELS = [("coarse", 0.03, 2), ("medium", 0.02, 4), ("fine", 0.015, 6)]
blade = load_blade(WINDIO)
z1 = blade_span_z(blade, R)


def fenics_time(yaml_path):
    """Pure FEniCS 2-D-solid boundary homogenization (compute_timo_boun) on the segment's
    end cross-section -- times the 2-D cross-sectional solve, not the 3-D segment."""
    from opensg.mesh.segment import SolidSegmentMesh
    from opensg.core.solid import compute_timo_boun
    d = os.path.dirname(yaml_path); cwd = os.getcwd(); os.chdir(d)
    try:
        sm = SolidSegmentMesh(os.path.abspath(yaml_path))
        mp, _den = sm.material_database
        t0 = time.time()
        out = compute_timo_boun(mp, sm.left_submesh)
        dt = time.time() - t0
        DL = np.asarray(out[0])
        ndof = 3 * sm.left_submesh["mesh"].topology.index_map(0).size_local
    finally:
        os.chdir(cwd)
    return dt, 0.5 * (DL + DL.T), ndof


rows = []
for name, ms, nr in LEVELS:
    cs = build_cross_section(blade, R, mesh_size=ms)
    # shell ring
    sp = os.path.join(SCR, "shell_%s.yaml" % name); emit_opensg_yaml(cs, sp)
    ring_in = load_ring(sp)
    t0 = time.time(); C6s = ring_6dof(ring_in); t_shell = time.time() - t0
    ndof_shell = 6 * len(ring_in["rx"])
    # prismatic hex segment
    res = hex_between_sections(cs, cs, z1, z1 + 2.0, nr=nr, nsp=4, nw=3, mesh_size=ms)
    oris, hmats = solid_yaml_payload(res, cs, cs)
    mat_names = sorted(set(hmats)); nix = {n: i for i, n in enumerate(mat_names)}
    mats = [{"name": n, **{k: _mat_block(blade, n)["elastic"][k] for k in ("E", "G", "nu")},
             "rho": _mat_block(blade, n)["density"]} for n in mat_names]
    yfe = os.path.join(SCR, "solid_%s.yaml" % name)
    write_solid_segment_yaml(yfe, res["nodes"], [list(h) for h in res["hexes"]],
                             list(hmats), np.asarray(oris), mats)
    # JAX solid (mixed hex + tet-web boundary)
    seg = dict(nodes=np.asarray(res["nodes"])[:, _PERM3],
               batches={"hex8": (np.asarray(res["hexes"]),
                                 np.array([nix[m] for m in hmats], int), np.asarray(oris)[:, _PERMF])},
               mat_param=np.array([_mp_row(blade, n) for n in mat_names]),
               nelem=len(res["hexes"]))
    web = np.array([t[0] == "web" for t in res["htag"]])
    seg = split_batches_to_tets(seg, mask=web)
    bL = extract_boundary_submesh(seg, "L")                 # JAX 2-D solid cross-section
    t0 = time.time(); DLj, _V0, _V1 = solve_boundary(bL); t_jax = time.time() - t0
    info = {"dof": 3 * len(bL["nodes"])}
    # FEniCS solid
    try:
        t_fe, DLf, ndof_fe = fenics_time(yfe)
    except Exception as e:
        t_fe, DLf, ndof_fe = float("nan"), np.full((6, 6), np.nan), 0
        print("  FEniCS failed:", str(e)[:120])
    rows.append((name, ndof_shell, t_shell, C6s[0, 0], info["dof"], t_jax, DLj[0, 0], ndof_fe, t_fe, DLf[0, 0]))
    print("%-7s | shell dof %-6d %5.2fs EA=%.3e | JAXsolid dof %-7d %6.2fs EA=%.3e | FEniCS dof %-7d %6.2fs EA=%.3e"
          % (name, ndof_shell, t_shell, C6s[0, 0], info["dof"], t_jax, DLj[0, 0], ndof_fe, t_fe, DLf[0, 0]), flush=True)

np.savez(os.path.join(OUT, "timing_r020.npz"),
         names=[r[0] for r in rows], rows=np.array([r[1:] for r in rows], float))
print("\nwrote", os.path.join(OUT, "timing_r020.npz"))
