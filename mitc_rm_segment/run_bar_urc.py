"""
run_bar_urc.py   [ Windows opensg_2_0_env ]
BAR-URC tapered SHELL segment -> MITC-RM Timoshenko 6x6 (+ origin), fully JAX-native:
  topological boundary extraction (no dolfinx) -> in-memory multi-material boundary
  Timoshenko -> 2-D MITC4 RM segment (axis-agnostic, per-element curvature) ->
  Dirichlet V0/V1 transfer -> 6x6.
Reference to compare: OpenSG-solid examples/3_get_beam_props_from_3D_solid_segment.py.
"""
import os, sys, json, time
import numpy as np
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from boundary_from_yaml import extract
from solve_segment_jax import _material_by_section, solve_boundary_bundle
from segment_element import assemble_segment, dirichlet_solve, build_C_Psi_segment, compute_k22
from opensg_jax.fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff

t0 = time.time()
seg_yaml = sys.argv[1]
tag = os.path.splitext(os.path.basename(seg_yaml))[0]
npz = os.path.join(HERE, "out", tag + ".npz")
extract(seg_yaml, npz)                                       # JAX-native topological extraction
b = np.load(npz, allow_pickle=True)
ax = int(b["axis"]); cross = tuple(j for j in range(3) if j != ax)
nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"]); subdom = np.asarray(b["seg_subdom"])
e1s, e2s, e3s = np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]), np.asarray(b["seg_e3"])
sections = json.loads(str(b["sections"])); materials = json.loads(str(b["materials"]))
t_extract = time.time() - t0

# origin = axial midpoint (matches the solid recentering); cross-section ref = global (0,0)
axial = nodes[:, ax]; origin_axial = 0.5 * (axial.min() + axial.max()); L = float(axial.max() - axial.min())

D_by, G_by = _material_by_section(sections, materials, center_ref=True)

# per-element hoop curvature (segment quads + each end contour)
k22_e = compute_k22(nodes[quads].mean(axis=1), e2s, e3s, quads)
def bnd_k22(side):
    rc = np.asarray(b["%s_cells" % side]); rx = np.asarray(b["%s_x" % side])
    return compute_k22(rx[rc].mean(axis=1), b["%s_e2" % side], b["%s_e3" % side], rc)

# boundary cross-sections solved in-memory
resL = solve_boundary_bundle(b, "L", shear="mitc", k22=bnd_k22("L"))
resR = solve_boundary_bundle(b, "R", shear="mitc", k22=bnd_k22("R"))

# segment assembly
Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment(nodes, quads, subdom, e1s, e2s, e3s, D_by, G_by, k22_e, cross=cross)
Dhe = np.asarray(Dhe); Dle = np.asarray(Dle)

def scatter(key):
    bd, bv = [], []
    for res, n2s in [(resL, np.asarray(b["L_node2seg"])), (resR, np.asarray(b["R_node2seg"]))]:
        Vv = res[key].reshape(-1, 5, 4)
        for i, sn in enumerate(n2s):
            for c in range(5):
                bd.append(5 * sn + c); bv.append(Vv[i, c, :])
    return np.array(bd), np.array(bv)

# V0 (EB) and V1 (Timoshenko) with boundary Dirichlet, no segment rigid constraints
bd0, bv0 = scatter("V0"); V0 = dirichlet_solve(Dhh, -Dhe, bd0, bv0)
C, Psi = build_C_Psi_segment(nodes, quads, cross=cross); Dc = C.T
bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
    jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle), jnp.array(Psi), jnp.array(Dc))
bb = np.asarray(bb); DhlV0 = np.asarray(DhlV0); DhlTV0Dle = np.asarray(DhlTV0Dle); V0DllV0 = np.asarray(V0DllV0)
bd1, bv1 = scatter("V1"); V1 = dirichlet_solve(Dhh, bb, bd1, bv1)
Deff = (np.asarray(Dee) + V0.T @ Dhe) / L
S, *_ = finalize_v1_and_compute_deff(
    jnp.array(V1), jnp.array(V0), jnp.array(Deff),
    jnp.array(V0DllV0 / L), jnp.array(DhlV0 / L), jnp.array(DhlTV0Dle / L), jnp.array(Psi), jnp.array(Dc))
S = np.asarray(0.5 * (np.asarray(S) + np.asarray(S).T))

print("\n=== BAR-URC tapered segment: %s ===" % tag)
print("beam axis = %s ; L = %.4f ; ORIGIN (axial) = %.6f" % ("xyz"[ax], L, origin_axial))
print("nodes %d, quads %d, layups %d ; L/R cross-section nodes %d/%d ; k22 range [%.3f, %.3f]"
      % (len(nodes), len(quads), len(sections), len(b["L_node2seg"]), len(b["R_node2seg"]),
         k22_e.min(), k22_e.max()))
print("\nTimoshenko 6x6 (order [ext, shear2, shear3, torsion, bend2, bend3]):")
for i in range(6):
    print("  " + "".join("%14.5e" % S[i, j] for j in range(6)))
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
print("\ndiagonal: " + "  ".join("%s=%.4e" % (LBL[i], S[i, i]) for i in range(6)))
print("\nextract %.1fs, total %.1fs" % (t_extract, time.time() - t0))
