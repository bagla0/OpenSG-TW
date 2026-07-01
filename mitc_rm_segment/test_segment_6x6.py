"""
test_segment_6x6.py   [ Windows opensg_2_0_env ]
Full Timoshenko 6x6 for the isotropic prismatic cylinder segment via the
boundary-YAML-first + Dirichlet-only flow:
  boundary rings -> V0,V1 (RM/MITC from the boundary YAML files);
  segment: Dirichlet V0 (RHS -Dhe) and Dirichlet V1 (RHS DhlTV0Dle - DhlV0);
  6x6 recovered with the same finalize (B_tim/C_tim/Q_tim/Ginv/Y_tim), per-length.
Checks: V0 AND V1 span-invariant, and segment 6x6 == ring 6x6 == analytic tube.
"""
import os, sys, json
import numpy as np
import jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from solve_segment_jax import build_material, solve_boundary_yaml, analytic_iso_tube
from segment_element import assemble_segment, dirichlet_solve, build_C_Psi_segment
from opensg_jax.fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff

npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "seg_iso_hR0.1_direct.npz")
tag = os.path.splitext(os.path.basename(npz))[0]; out_dir = os.path.dirname(npz)
b = np.load(npz, allow_pickle=True)
mat = json.loads(str(b["materials"]))[0]["elastic"]; E, nu = float(mat["E"][0]), float(mat["nu"][0])
D, G, t = build_material(b, center_ref=True)
nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"]); subdom = np.asarray(b["seg_subdom"])
R = float(np.mean(np.hypot(nodes[:, 1], nodes[:, 2]))); k22 = -1.0 / R
NC = len(b["L_x"]); Nn = len(nodes); NL = Nn // NC - 1
L = float(nodes[:, 0].max() - nodes[:, 0].min())

resL = solve_boundary_yaml(os.path.join(out_dir, "boundary_%s_L.yaml" % tag), shear="mitc")
resR = solve_boundary_yaml(os.path.join(out_dir, "boundary_%s_R.yaml" % tag), shear="mitc")
Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment(
    nodes, quads, subdom, b["seg_e1"], b["seg_e2"], b["seg_e3"], {0: D}, {0: G}, {0: k22})
Dhe = np.asarray(Dhe); Dle = np.asarray(Dle)


def scatter(key):
    """boundary field `key` (V0/V1) -> (bdofs, bvals) on segment DOFs."""
    bd, bv = [], []
    for res, n2s in [(resL, np.asarray(b["L_node2seg"])), (resR, np.asarray(b["R_node2seg"]))]:
        Vv = res[key].reshape(-1, 5, 4)
        for i, sn in enumerate(n2s):
            for c in range(5):
                bd.append(5 * sn + c); bv.append(Vv[i, c, :])
    return np.array(bd), np.array(bv)


def span(V):
    Vn = V.reshape(Nn, 5, 4); rm = np.max(np.abs(V))
    mv = max(max(np.max(np.abs(Vn[j*NC + k] - Vn[k])) for k in range(NC)) for j in range(NL + 1))
    return mv / rm


# rigid kernel + constraints (used to PROJECT the V1 RHS, as the 1-D solve does)
C, Psi = build_C_Psi_segment(nodes, quads); Dc = C.T

# --- V0 (EB) ---
bd0, bv0 = scatter("V0")
V0 = dirichlet_solve(Dhh, -Dhe, bd0, bv0)
# --- V1 (Timoshenko): projected RHS (matches the ring's system) + boundary V1 Dirichlet ---
bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
    jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle), jnp.array(Psi), jnp.array(Dc))
bb = np.asarray(bb); DhlV0 = np.asarray(DhlV0); DhlTV0Dle = np.asarray(DhlTV0Dle); V0DllV0 = np.asarray(V0DllV0)
bd1, bv1 = scatter("V1")
V1 = dirichlet_solve(Dhh, bb, bd1, bv1)

print("V0 span-invariance rel = %.2e | V1 span-invariance rel = %.2e" % (span(V0), span(V1)))

# --- 6x6 recovery (per unit length) ---
Deff = (np.asarray(Dee) + V0.T @ Dhe) / L
S, *_ = finalize_v1_and_compute_deff(
    jnp.array(V1), jnp.array(V0), jnp.array(Deff),
    jnp.array(V0DllV0 / L), jnp.array(DhlV0 / L), jnp.array(DhlTV0Dle / L),
    jnp.array(Psi), jnp.array(Dc))
S = np.asarray(0.5 * (np.asarray(S) + np.asarray(S).T))

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
ana = analytic_iso_tube(R, t, E, nu); ring = np.diag(resL["C6"])
print("\nTimoshenko 6x6 diagonal:")
print("%-5s %14s %14s %14s %8s" % ("term", "segment", "ring", "analytic", "%err"))
for i, k in enumerate(LBL):
    print("%-5s %14.4e %14.4e %14.4e %+7.1f%%"
          % (k, S[i, i], ring[i], ana[k], 100 * (S[i, i] - ana[k]) / ana[k]))
print("max |segment 6x6 - ring 6x6| / max|ring| = %.2e"
      % (np.max(np.abs(S - resL["C6"])) / np.max(np.abs(resL["C6"]))))
