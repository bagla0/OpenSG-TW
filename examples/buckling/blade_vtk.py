"""blade_vtk.py -- export the blade shell mesh + buckling mode shapes (JAX-FEA & RM-OpenSG) + membrane N
as PolyData VTK for ParaView rendering.  Reads data/blade_mesh.npz, blade_fea.npz, blade_rm.npz."""
import os, numpy as np, pyvista as pv
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VTK = os.path.join(D, "vtk"); os.makedirs(VTK, exist_ok=True)

m = np.load(os.path.join(D, "blade_mesh.npz"))
nodes, quads = m["nodes"], m["quads"]
faces = np.hstack([np.full((len(quads), 1), 4), quads]).astype(np.int64).ravel()
NM = 4                                                        # modes to export


def mode_scale(nodes, modes, k):
    u = modes[:, :3, k]
    return 0.06 * (nodes[:, 0].max() - nodes[:, 0].min()) / (np.abs(u).max() + 1e-30)


for tag in ["fea", "rm"]:
    fp = os.path.join(D, "blade_%s.npz" % tag)
    if not os.path.exists(fp):
        print("skip", tag); continue
    z = np.load(fp); modes = z["modes"]; loads = z["loads"]; Nv = z["Nvec"]
    base = pv.PolyData(nodes, faces)
    base.cell_data["N11"] = Nv[:, 0]; base.cell_data["N22"] = Nv[:, 1]; base.cell_data["N12"] = Nv[:, 2]
    base.save(os.path.join(VTK, "blade_%s_N.vtk" % tag))
    for k in range(min(NM, modes.shape[2])):
        s = mode_scale(nodes, modes, k)
        dp = pv.PolyData(nodes + s * modes[:, :3, k], faces)
        dp.point_data["umag"] = np.linalg.norm(modes[:, :3, k], axis=1)
        dp.save(os.path.join(VTK, "blade_%s_mode%d.vtk" % (tag, k + 1)))
    print("%s: N + %d modes (lambda_1=%.4f)" % (tag, min(NM, modes.shape[2]), loads[0]))
print("wrote", VTK)
