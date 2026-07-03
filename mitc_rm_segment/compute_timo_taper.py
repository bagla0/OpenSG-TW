"""
compute_timo_taper.py    [ Windows opensg_2_0_env ]
========================================================================
Tapered-segment homogenization from the surface-quad MITC-RM shell element
(the 8-strain operator of segment_element._quad_ops), boundary-V0-Dirichlet.

    compute_timo_taper(bundle) -> {"EB": 4x4, ...}

EB (Euler-Bernoulli 4x4, per unit length) FIRST -- the benchmark the user asked
for: it must equal the FEniCS SOLID tapered-segment EB (opensg.core.solid.
compute_stiffness, same origin).  For a PRISMATIC cylinder it must also equal the
1-D cross-section EB (== analytic tube).  The Timoshenko 6x6 (V1 step) is wired in
the same structure and returned when return_timo=True (TODO: shear recovery).

Ordering (Dee = _macro_BD, eb = [g11, k1, k2, k3]):  [EA, GJ, EI2, EI3].
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax.numpy as jnp
from segment_element import assemble_segment, dirichlet_solve, compute_k22, build_C_Psi_segment
from solve_segment_jax import _material_by_section, solve_boundary_bundle
from opensg_jax.fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff


def _seg_k22(nodes, quads, e2s, e3s, cross, mode, R=None):
    if mode == "tube":
        c = nodes[:, cross].mean(0)
        if R is None:
            R = float(np.mean(np.hypot(nodes[:, cross[0]] - c[0], nodes[:, cross[1]] - c[1])))
        return np.full(len(quads), -1.0 / R), R
    cent = nodes[quads].mean(1)                       # per-quad centroid
    return compute_k22(cent, np.asarray(e2s), np.asarray(e3s), quads), R


def compute_timo_taper(bundle, center_ref=True, shear="mitc", k22_mode="tube",
                       return_timo=False, full_curvature=False, verbose=False):
    b = bundle
    ax = int(b["axis"]); cross = tuple(j for j in range(3) if j != ax)
    nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"]); subdom = np.asarray(b["seg_subdom"])
    e1s, e2s, e3s = np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]), np.asarray(b["seg_e3"])
    sections = json.loads(str(b["sections"])); materials = json.loads(str(b["materials"]))
    D_by, G_by = _material_by_section(sections, materials, center_ref)
    k22_e, R = _seg_k22(nodes, quads, e2s, e3s, cross, k22_mode)

    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment(
        nodes, quads, subdom, e1s, e2s, e3s, D_by, G_by, k22_e, cross, full_curvature)

    # boundary rings -> V0 (4 EB modes), scattered as Dirichlet BCs
    # (k22_mode='general': per-edge geometric hoop curvature on the rings too,
    #  from the parent-quad frames -- same convention as the segment quads)
    def _bnd_k22(side):
        if k22_mode != "general":
            return None
        rc = np.asarray(b["%s_cells" % side]); rx = np.asarray(b["%s_x" % side])
        return compute_k22(rx[rc].mean(axis=1), np.asarray(b["%s_e2" % side]),
                           np.asarray(b["%s_e3" % side]), rc)
    resL = solve_boundary_bundle(b, "L", center_ref, shear, k22=_bnd_k22("L"))
    resR = solve_boundary_bundle(b, "R", center_ref, shear, k22=_bnd_k22("R"))
    bdofs, bvals = [], []
    for res, n2s in [(resL, np.asarray(b["L_node2seg"])), (resR, np.asarray(b["R_node2seg"]))]:
        V0 = res["V0"].reshape(-1, 5, 4)
        for i, sn in enumerate(n2s):
            for c in range(5):
                bdofs.append(5 * int(sn) + c); bvals.append(V0[i, c, :])
    bdofs = np.array(bdofs); bvals = np.array(bvals, float)
    V0seg = dirichlet_solve(np.asarray(Dhh), -np.asarray(Dhe), bdofs, bvals)

    L = float(nodes[:, ax].max() - nodes[:, ax].min())
    EB = (np.asarray(Dee) + V0seg.T @ np.asarray(Dhe)) / L          # 4x4 per unit length
    out = {"EB": EB, "L": L, "R": R, "V0seg": V0seg, "ring_C6": resL["C6"],
           "origin": float(nodes[:, ax].mean())}

    if return_timo:                                                # full Timoshenko 6x6 (V1 step)
        Dhh_a, Dhl_a, Dll_a, Dle_a = (np.asarray(Dhh), np.asarray(Dhl),
                                      np.asarray(Dll), np.asarray(Dle))
        C, Psi = build_C_Psi_segment(nodes, quads, cross); Dc = C.T
        bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
            jnp.array(V0seg), jnp.array(Dhl_a), jnp.array(Dll_a), jnp.array(Dle_a),
            jnp.array(Psi), jnp.array(Dc))
        bb = np.asarray(bb)
        bd1, bv1 = [], []
        for res, n2s in [(resL, np.asarray(b["L_node2seg"])), (resR, np.asarray(b["R_node2seg"]))]:
            V1 = res["V1"].reshape(-1, 5, 4)
            for i, sn in enumerate(n2s):
                for c in range(5):
                    bd1.append(5 * int(sn) + c); bv1.append(V1[i, c, :])
        V1seg = dirichlet_solve(Dhh_a, bb, np.array(bd1), np.array(bv1, float))
        S, *_ = finalize_v1_and_compute_deff(
            jnp.array(V1seg), jnp.array(V0seg), jnp.array(EB),
            jnp.array(np.asarray(V0DllV0) / L), jnp.array(np.asarray(DhlV0) / L),
            jnp.array(np.asarray(DhlTV0Dle) / L), jnp.array(Psi), jnp.array(Dc))
        S = np.asarray(S); out["C6"] = 0.5 * (S + S.T)

    if verbose:
        d = np.diag(EB)
        print("EB 4x4 [EA, GJ, EI2, EI3] diag:", np.array2string(d, precision=4))
        if "C6" in out:
            print("Timo 6x6 [EA,GA2,GA3,GJ,EI2,EI3] diag:",
                  np.array2string(np.diag(out["C6"]), precision=4))
    return out


def compute_timo_taper_jax(bundle, center_ref=True, shear="mitc", k22_mode="tube"):
    """JAX-assembled EB path (segment_element_jax.assemble_segment_jax) -- same
    result as compute_timo_taper, jit+vmap over quads.  Boundary V0 + the small
    Dirichlet solve stay in NumPy (already fast)."""
    from segment_element_jax import assemble_segment_jax
    b = bundle
    ax = int(b["axis"]); cross = tuple(j for j in range(3) if j != ax)
    nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"]); subdom = np.asarray(b["seg_subdom"])
    e1s, e2s, e3s = np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]), np.asarray(b["seg_e3"])
    sections = json.loads(str(b["sections"])); materials = json.loads(str(b["materials"]))
    D_by, G_by = _material_by_section(sections, materials, center_ref)
    nsec = len(D_by)
    D_stack = np.array([D_by[i] for i in range(nsec)])
    G_stack = np.array([G_by[i] for i in range(nsec)])
    k22_e, R = _seg_k22(nodes, quads, e2s, e3s, cross, k22_mode)
    Dhh, Dhe, Dee = assemble_segment_jax(nodes, quads, subdom, e1s, e2s, e3s, D_stack, G_stack, k22_e, cross)
    Dhh, Dhe, Dee = np.asarray(Dhh), np.asarray(Dhe), np.asarray(Dee)

    resL = solve_boundary_bundle(b, "L", center_ref, shear)
    resR = solve_boundary_bundle(b, "R", center_ref, shear)
    bdofs, bvals = [], []
    for res, n2s in [(resL, np.asarray(b["L_node2seg"])), (resR, np.asarray(b["R_node2seg"]))]:
        V0 = res["V0"].reshape(-1, 5, 4)
        for i, sn in enumerate(n2s):
            for c in range(5):
                bdofs.append(5 * int(sn) + c); bvals.append(V0[i, c, :])
    bdofs = np.array(bdofs); bvals = np.array(bvals, float)
    V0seg = dirichlet_solve(Dhh, -Dhe, bdofs, bvals)
    L = float(nodes[:, ax].max() - nodes[:, ax].min())
    EB = (Dee + V0seg.T @ Dhe) / L
    return {"EB": EB, "L": L, "R": R, "origin": float(nodes[:, ax].mean())}


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "seg_iso_hR0.1_direct.npz")
    b = np.load(npz, allow_pickle=True)
    r = compute_timo_taper(b, verbose=True)
    EB = r["EB"]; R = r["R"]
    mat = json.loads(str(b["materials"]))[0]["elastic"]; E, nu = float(mat["E"][0]), float(mat["nu"][0])
    sec = json.loads(str(b["sections"]))[0]["layup"]; t = float(sum(p[1] for p in sec))
    Gs = E / (2 * (1 + nu)); A = 2 * np.pi * R * t; I = np.pi * R**3 * t; J = 2 * np.pi * R**3 * t
    ana = [E * A, Gs * J, E * I, E * I]; LBL = ["EA", "GJ", "EI2", "EI3"]
    print("\nsegment EB vs analytic isotropic tube  (R=%.3f  t=%.3f  h/R=%.2f  L=%.3f  origin=%.3f):"
          % (R, t, t / R, r["L"], r["origin"]))
    print("  %-5s %14s %14s %9s" % ("term", "segment EB", "analytic", "%err"))
    for i in range(4):
        print("  %-5s %14.4e %14.4e %+8.2f%%" % (LBL[i], EB[i, i], ana[i], 100 * (EB[i, i] - ana[i]) / ana[i]))
    print("ring 6x6 diag:", np.array2string(np.diag(r["ring_C6"]), precision=3))
