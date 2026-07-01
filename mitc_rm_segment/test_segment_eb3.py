"""
test_segment_eb3.py   [ Windows opensg_2_0_env ]
EB span-invariance, boundary-YAML-first + Dirichlet-only segment (user's flow):
  1. read each end ring from its standalone 1-D boundary YAML file, solve it
     (RM/MITC, shear='mitc' -- consistent with the 2-D MITC4 at span-invariance);
  2. assemble the 2-D segment; impose the boundary V0 as DIRICHLET (no rigid-body
     constraints -- the boundary nodes fix all 6 rigid modes);
  3. check V0 is the same at each node corresponding to the boundary (structured
     ring-to-ring), and the EB 4x4 vs the closed-form isotropic tube.
Boundary YAMLs are produced by boundary_from_yaml.py (run it first).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from solve_segment_jax import build_material, solve_boundary_yaml
from segment_element import assemble_segment, dirichlet_solve

npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "seg_iso_hR0.1_direct.npz")
tag = os.path.splitext(os.path.basename(npz))[0]
out_dir = os.path.dirname(npz)
b = np.load(npz, allow_pickle=True)
mat = json.loads(str(b["materials"]))[0]["elastic"]; E, nu = float(mat["E"][0]), float(mat["nu"][0])
D, G, t = build_material(b, center_ref=True)
nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"]); subdom = np.asarray(b["seg_subdom"])
R = float(np.mean(np.hypot(nodes[:, 1], nodes[:, 2]))); k22 = -1.0 / R
NC = len(b["L_x"]); Nn = len(nodes); NL = Nn // NC - 1

# 1. boundary rings from their YAML files
resL = solve_boundary_yaml(os.path.join(out_dir, "boundary_%s_L.yaml" % tag), shear="mitc")
resR = solve_boundary_yaml(os.path.join(out_dir, "boundary_%s_R.yaml" % tag), shear="mitc")
print("boundary YAML solved. ring 6x6 diag:", np.array2string(np.diag(resL["C6"]), precision=3))

# 2. assemble segment + Dirichlet transfer (V0[i] -> segment node node2seg[i]; order preserved)
Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment(
    nodes, quads, subdom, b["seg_e1"], b["seg_e2"], b["seg_e3"], {0: D}, {0: G}, {0: k22})
bdofs, bvals = [], []
for res, n2s in [(resL, np.asarray(b["L_node2seg"])), (resR, np.asarray(b["R_node2seg"]))]:
    V0 = res["V0"].reshape(-1, 5, 4)
    for i, sn in enumerate(n2s):
        for c in range(5):
            bdofs.append(5 * sn + c); bvals.append(V0[i, c, :])
bdofs = np.array(bdofs); bvals = np.array(bvals)
V0seg = dirichlet_solve(Dhh, -np.asarray(Dhe), bdofs, bvals)

# 3a. span-invariance: node = j*NC + k
V0n = V0seg.reshape(Nn, 5, 4); refmax = np.max(np.abs(V0seg))
per_ring = [max(np.max(np.abs(V0n[j*NC + k] - V0n[k])) for k in range(NC)) for j in range(NL + 1)]
maxvar = max(per_ring)
print("per-ring max|V0[ring j]-V0[ring 0]|:", np.array2string(np.array(per_ring), precision=3))
print("SPAN-INVARIANCE rel = %.2e -> %s" % (maxvar / refmax, "PASS" if maxvar / refmax < 1e-3 else "CHECK"))

# 3b. EB 4x4 (per unit length) vs closed-form tube
L = float(nodes[:, 0].max() - nodes[:, 0].min())
Deff = (np.asarray(Dee) + V0seg.T @ np.asarray(Dhe)) / L
Gs = E/(2*(1+nu)); A = 2*np.pi*R*t; I = np.pi*R**3*t; J = 2*np.pi*R**3*t
ana = [E*A, Gs*J, E*I, E*I]; LBL = ["EA", "GJ", "EI2", "EI3"]
print("EB 4x4 diagonal vs analytic:")
for i in range(4):
    print("  %-4s %14.4e  %14.4e  %+7.1f%%" % (LBL[i], Deff[i, i], ana[i], 100*(Deff[i, i]-ana[i])/ana[i]))
