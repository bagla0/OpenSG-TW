"""
solve_segment_jax.py    [ run in the Windows opensg_2_0_env ]
========================================================================
STAGE 3 of the hybrid MITC-RM tapered-segment pipeline (isotropic first).

    stage 1  make_cylinder_segment.py        -> surface-quad YAML
    stage 2  extract_boundaries_dolfinx.py   -> boundary rings + maps (.npz)
    stage 3  solve_segment_jax.py            -> RM/MITC V0,V1 + 6x6      <-- THIS FILE

PLAN
----
  PART 1  (implemented, self-validating)
      Solve each END RING as a 1-D Reissner-Mindlin / MITC cross-section SG,
      reusing the VALIDATED opensg_jax RM assembly (msg_rm_timo.timoshenko_rm
      with return_warp=True).  This yields the boundary warping fields
      V0 (4 Euler-Bernoulli modes) and V1 (Timoshenko shear-warping), plus the
      ring Timoshenko 6x6.  We check the ring 6x6 against the closed-form
      isotropic tube (EA, EI, GJ) so the material/geometry wiring is trusted
      before it feeds the segment.

  PART 2  (next: the one genuinely new element)
      A 2-D MITC4 Reissner-Mindlin SHELL element on the curved quad surface
      (5 DOF/node [w1,w2,w3,w1,w2]; bilinear N(xi,eta); 2x2 surface Jacobian;
      transverse-shear tying per Chapelle-Bathe "The Finite Element Analysis of
      Shells" sec 8.2 / plate sec 7.2.4).  Then scatter the ring V0/V1 onto the
      segment boundary DOFs as Dirichlet BCs, solve the segment, and (the check
      you asked for) confirm the segment warping at ANY interior ring equals the
      boundary warping node-for-node -- the prismatic span-invariance identity.

Only the isotropic case is exercised here (per request).
"""

import os
import sys
import json
import numpy as np

# --- make the opensg_jax package importable -------------------------------------
REPO = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from opensg_jax.fe_jax.msg_materials import compute_ABD_matrix, shift_abd_reference
from opensg_jax.fe_jax.msg_transverse_shear import transverse_shear_stiffness
from opensg_jax.fe_jax.msg_rm_timo import timoshenko_rm     # 1-D RM solve (return_warp -> V0,V1)

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


# ============================================================ material (ABD + G)
def build_material(bundle, center_ref=True):
    """Return (D 6x6 ABD, G 2x2 transverse shear, t) for the (single) layup.

    The mesh sits on the wall MID-surface, so the ABD -- which compute_ABD_matrix
    references to the bottom face -- is parallel-axis shifted by t/2 to the centre
    (shift_abd_reference); the transverse-shear block G is reference-independent.
    """
    materials = json.loads(str(bundle["materials"]))
    sections = json.loads(str(bundle["sections"]))
    mat_db = {m["name"]: {"E": m["elastic"]["E"], "G": m["elastic"]["G"],
                          "nu": m["elastic"]["nu"]} for m in materials}
    layup = sections[0]["layup"]                       # [[mat, ply_t, angle], ...]
    mat_names = [ply[0] for ply in layup]
    thick = [float(ply[1]) for ply in layup]
    angles = [float(ply[2]) for ply in layup]
    t = float(sum(thick))

    D = np.asarray(compute_ABD_matrix(thick, angles, mat_names, mat_db)[0])   # 6x6, bottom-ref
    if center_ref:
        D = shift_abd_reference(D, t / 2.0)                                    # -> mid-wall
    G = np.asarray(transverse_shear_stiffness(thick, angles, mat_names, mat_db)[0])  # 2x2
    return D, G, t


# ================================================== PART 1: 1-D boundary ring SG
def order_ring(ring_x):
    """Sort a ring's nodes by hoop angle and return (nodes2d(NC,2), elems(NC,2),
    perm, R).  The dolfinx submesh nodes are unordered; sorting into a consistent
    CCW loop matches the 1-D generator convention (uniform inward normal), so the
    reused element operators see the same orientation they were validated with.
    `perm[i]` = original ring-node index of sorted node i (kept so ring warping
    can later be matched to segment boundary nodes by hoop position).
    """
    y, z = ring_x[:, 1], ring_x[:, 2]                  # cross-section plane = (y, z)
    th = np.arctan2(z, y)
    perm = np.argsort(th)
    nodes2d = np.column_stack([y[perm], z[perm]])
    NC = len(perm)
    elems = np.array([[i, (i + 1) % NC] for i in range(NC)], dtype=int)   # closed loop
    R = float(np.mean(np.hypot(y, z)))
    return nodes2d, elems, perm, R


def solve_ring(ring_x, D, G, shear="mitc_both"):
    """Solve the ring as a 1-D RM/MITC cross-section SG.

    Returns dict with the Timoshenko 6x6 and the warping fields V0 (ndof,4) and
    V1 (ndof,4) in the ring's SORTED node order (5 DOF/node [w1,w2,w3,w1r,w2r]),
    plus `perm` and `R`.  k22 = -1/R is the (uniform) hoop curvature of the tube
    in the code's -1/R convention; the sign is confirmed by the 6x6 check below.
    """
    nodes2d, elems, perm, R = order_ring(ring_x)
    lpe = np.zeros(len(elems), dtype=int)
    D_by = {0: D}; G_by = {0: G}
    k22 = np.full(len(elems), -1.0 / R)
    C6, Deff, V0, V1 = timoshenko_rm(nodes2d, elems, lpe, D_by, G_by, k22,
                                     p=1, return_warp=True, shear=shear)
    return {"C6": np.asarray(C6), "V0": np.asarray(V0), "V1": np.asarray(V1),
            "perm": perm, "R": R, "nodes2d": nodes2d, "elems": elems}


def analytic_iso_tube(R, t, E, nu):
    """Closed-form thin isotropic-tube Timoshenko diagonal (mid-wall reference)."""
    Gs = E / (2.0 * (1.0 + nu))
    A = 2.0 * np.pi * R * t
    I = np.pi * R**3 * t                 # bending 2nd moment (thin ring)
    J = 2.0 * np.pi * R**3 * t           # torsion constant (thin tube)
    ks = 0.5                             # thin-ring shear coefficient (approx)
    return dict(EA=E * A, GA2=Gs * A * ks, GA3=Gs * A * ks, GJ=Gs * J,
                EI2=E * I, EI3=E * I)


# ================================================== PART 2: 2-D MITC-RM segment
# TODO (next): assemble the segment RM fluctuation system on the quad surface.
#   * bilinear shape functions N(xi,eta) on [-1,1]^2, 2x2 Gauss for the D-energy;
#   * surface Jacobian J = [dX/dxi, dX/deta] (3x2) -> in-plane derivatives d/dx1
#     (axial, .e1) and d/ds (hoop, .e2); e1/e2/e3 come from seg_e1/e2/e3;
#   * D-strain rows [eps11, eps22, 2eps12, k11, k22, 2k12] extend the 1-D BDq by
#     adding the AXIAL (xi1) derivatives (zero for a span-invariant field);
#   * MITC4 transverse-shear tying (Dvorkin-Bathe / Chapelle-Bathe sec 8.2):
#       gamma_xi_zeta  tied at (xi=0, eta=+/-1), linear in eta;
#       gamma_eta_zeta tied at (xi=+/-1, eta=0), linear in xi;
#   * reuse msg_solver (assemble_kkt / solve_fluctuation_field / prepare_v1_rhs /
#     finalize_v1_and_compute_deff) unchanged -- it is element-agnostic.
# Then: Dirichlet the boundary DOFs to the ring V0/V1 (map ring node -> segment
# node via L_node2seg[perm]), solve the segment, and assert
#   V0_seg[interior ring node] == V0_ring[same hoop node]  (prismatic invariance).
def assemble_segment_rm(bundle, D, G):
    raise NotImplementedError("PART 2 -- 2-D MITC4 RM segment element (next step)")


# ============================================================================ run
def main(npz_path):
    b = np.load(npz_path, allow_pickle=True)
    D, G, t = build_material(b, center_ref=True)
    # isotropic engineering constants (for the closed-form check)
    mat = json.loads(str(b["materials"]))[0]["elastic"]
    E, nu = float(mat["E"][0]), float(mat["nu"][0])

    print("=== PART 1: boundary-ring RM/MITC 6x6 (isotropic) ===")
    print("material: E=%.3e  nu=%.2f   wall t=%.4f" % (E, nu, t))
    left = solve_ring(b["L_x"], D, G)
    right = solve_ring(b["R_x"], D, G)
    R = left["R"]
    ana = analytic_iso_tube(R, t, E, nu)

    C6 = left["C6"]; d = np.diag(C6)
    print("ring radius R=%.4f  h/R=%.3f" % (R, t / R))
    print("%-5s %14s %14s %10s" % ("term", "ring 6x6", "analytic", "%err"))
    for i, k in enumerate(LBL):
        err = 100.0 * (d[i] - ana[k]) / ana[k]
        print("%-5s %14.4e %14.4e %+9.1f%%" % (k, d[i], ana[k], err))

    # left/right must be identical for a prismatic segment (same cross-section)
    dLR = np.max(np.abs(left["C6"] - right["C6"])) / max(1.0, np.max(np.abs(left["C6"])))
    print("left-vs-right 6x6 max rel diff: %.2e (expect ~0, both ends identical)" % dLR)
    print("V0 shape", left["V0"].shape, " V1 shape", left["V1"].shape,
          " (ndof = 5 x %d ring nodes)" % (left["V0"].shape[0] // 5))
    print("\nPART 2 (2-D MITC4 RM segment + V0/V1 span-invariance check): next.")
    return left, right


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out", "seg_iso_hR0.1.npz")
    main(npz)
