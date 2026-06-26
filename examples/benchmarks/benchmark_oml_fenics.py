"""
FEniCS solid OML 3D-stress for the MSG-TW benchmark (run in WSL, dolfinx 0.8.0).

Uses the corrected training-data opensg-FEniCS package: SolidBounMesh +
compute_timo_boun (KKT homogenization) + stress_recov.local_strain (corrected
recov chain, FF passed at the calling level).  Applies the SAME common beam
force FF as the JAX-TW run, builds the GLOBAL/beam-frame 3D stress field
(sigma_global = Rsig @ sigma_material) on CG1, and writes every nodal
(y2, y3, S11..S12) so the comparison script can pick the OML nodes by tolerance.

Output: <outdir>/oml_fenics.txt
"""
import os
import sys
import numpy as np
import dolfinx, basix, ufl
from mpi4py import MPI
from dolfinx.fem import Function, Expression, functionspace

from opensg.mesh.segment import SolidBounMesh
from opensg.core.solid import compute_timo_boun
import opensg.core.stress_recov as stress_recov
from opensg.utils import solid as su

# ---- COMMON beam force (VABS [F1,F2,F3,M1,M2,M3]) — identical to JAX run ----
FF = np.array([1.0e5, 5.0e4, 5.0e4, 5.0e4, 1.0e5, 1.0e5])

MESH_YAML = "/mnt/c/Users/bagla0/OpenSG/examples/data/Solid_2DSG/2Dsolid_0.yaml"
OUT = sys.argv[1] if len(sys.argv) > 1 else "oml_fenics.txt"
SEGID = 0

# dummy beam reactions (local_strain still indexes beam_out, but FF overrides)
dummy = [[["", 0.0] for _ in range(6)] for _ in range(5)]
beam_out = (dummy, dummy)

sm = SolidBounMesh(MESH_YAML)
material_parameters, density = sm.material_database
meshdata = sm.meshdata
mesh = meshdata["mesh"]

timo = compute_timo_boun(material_parameters, meshdata)
print("Deff_srt diag:", np.diag(timo[0]))

st_3D_m, u_loc, strain_quad, stress_quad, coord_quad = stress_recov.local_strain(
    timo, beam_out, SEGID, meshdata, material_parameters, FF)

# ---- GLOBAL/beam-frame stress = Rsig(frame) @ (C_material @ st_3D_m) ----
CC_ = su.CC(material_parameters)
V_stiff = functionspace(mesh, basix.ufl.element("DG", mesh.topology.cell_name(), 0, shape=(6, 6)))
stiff = Function(V_stiff)
for i, subb in enumerate(meshdata["subdomains"].values):
    stiff.x.array[36 * i:36 * i + 36] = CC_[subb].flatten()
stress_mat = ufl.dot(stiff, st_3D_m)
stress_glob = ufl.dot(su.Rsig(meshdata["frame"]), stress_mat)
hdr = ("FF=" + ",".join(f"{v:g}" for v in FF) + "\n"
       "y2 y3 S11 S22 S33 S23 S13 S12 (global frame, Pa)")


def save(space, path, tag):
    sg = Function(space)
    sg.interpolate(Expression(stress_glob, space.element.interpolation_points(),
                              comm=MPI.COMM_WORLD))
    coords = space.tabulate_dof_coordinates()       # (N,3) = (0, y2, y3)
    stress = sg.x.array.reshape(-1, 6)              # [S11,S22,S33,S23,S13,S12]
    np.savetxt(path, np.column_stack([coords[:, 1], coords[:, 2], stress]),
               header=hdr, fmt="%18.8e")
    print(f"wrote {path}  ({len(stress)} {tag} points)")
    return sg


# (1) PRIMARY: Gauss quadrature, degree 2 (native dehom recovery points)
Qg = functionspace(mesh, basix.ufl.quadrature_element(
    mesh.basix_cell(), value_shape=(6,), degree=2, scheme="default"))
save(Qg, OUT.replace(".txt", "_gauss.txt"), "gauss-q2")

# (2) ADDITIONAL: CG2 (quadratic Lagrange) interpolation, for comparison
Vc = functionspace(mesh, basix.ufl.element("CG", mesh.topology.cell_name(), 2, shape=(6,)))
sg_cg2 = save(Vc, OUT.replace(".txt", "_cg2.txt"), "CG2")

# (3) EXACT-PATH: evaluate at an external set of (y2,y3) coordinates (the TW
# reference-mesh path), so TW and solid report at the SAME coordinates.
import dolfinx.geometry as geom
pc = os.path.join(os.path.dirname(OUT), "oml_path_coords.txt")
if os.path.exists(pc):
    P = np.loadtxt(pc).reshape(-1, 2)               # (y2, y3); already nudged inside
    pts = np.zeros((len(P), 3)); pts[:, 1:3] = P    # eval at the exact path coords
    bb = geom.bb_tree(mesh, mesh.topology.dim)
    coll = geom.compute_colliding_cells(
        mesh, geom.compute_collisions_points(bb, pts), pts)
    keep, cells = [], []
    for i in range(len(pts)):
        lk = coll.links(i)
        if len(lk) > 0:
            keep.append(i); cells.append(lk[0])
    vals = sg_cg2.eval(pts[keep], np.array(cells, dtype=np.int32))
    out_atpath = np.column_stack([P[keep, 0], P[keep, 1], vals])
    ap = os.path.join(os.path.dirname(OUT), "oml_fenics_atpath.txt")
    np.savetxt(ap, out_atpath, header=hdr, fmt="%18.8e")
    print(f"wrote {ap}  ({len(keep)}/{len(P)} path coords evaluated)")
