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
    """The CURVATURE argument: 'tube' = uniform -1/R; 'general' = per-element
    geometric (compute_k22: median + flat-snap-to-0, bounded); 'zero' = off."""
    if mode == "zero":
        return np.zeros(len(quads)), R
    if mode == "tube":
        c = nodes[:, cross].mean(0)
        if R is None:
            R = float(np.mean(np.hypot(nodes[:, cross[0]] - c[0], nodes[:, cross[1]] - c[1])))
        return np.full(len(quads), -1.0 / R), R
    cent = nodes[quads].mean(1)                       # per-quad centroid
    return compute_k22(cent, np.asarray(e2s), np.asarray(e3s), quads), R


def compute_timo_taper(bundle, center_ref=True, shear="mitc_both", k22_mode="tube",
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
        if k22_mode == "zero":
            return "zero"
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


def compute_timo_taper_jax(bundle, center_ref=True, shear="mitc_both", k22_mode="tube"):
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


def compute_timo_taper_layup(bundle, bundle_next, center_ref=False, shear="mitc_both",
                             k22_mode="general", verbose=False):
    """Layup-TAPERED segment: the OpenSG shell mesh freezes each segment's laminate
    at its INBOARD (left) reference, so a segment's RIGHT boundary carries the
    inboard-frozen (too-thick) layup -- NOT the true cross-section at that station
    (see ref_bar_urc_shell_spar_mislabel).  Because consecutive segments share the
    face (seg N's R geometry == seg N+1's L geometry, identical connectivity), the
    true outboard layup is seg N+1's LEFT layup.  This routine blends the per-quad
    ABD/G linearly from seg N's layup (at x_min) to seg N+1's layup (at x_max) by
    span fraction, and uses seg N+1's LEFT ring as the RIGHT boundary -- so BOTH
    end cross-sections are one-to-one with the 3-D solid.

    `bundle` = segment N, `bundle_next` = segment N+1 (same numEl mesh; must have
    identical connectivity, checked).  Returns the Timoshenko 6x6 + EB.
    """
    b, bn = bundle, bundle_next
    ax = int(b["axis"]); cross = tuple(j for j in range(3) if j != ax)
    nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"])
    e1s, e2s, e3s = np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]), np.asarray(b["seg_e3"])
    sdN = np.asarray(b["seg_subdom"]); sdN1 = np.asarray(bn["seg_subdom"])
    if not np.array_equal(quads, np.asarray(bn["seg_cells"])):
        raise ValueError("segments N and N+1 must share connectivity for layup taper")
    DN, GN = _material_by_section(json.loads(str(b["sections"])), json.loads(str(b["materials"])), center_ref)
    DN1, GN1 = _material_by_section(json.loads(str(bn["sections"])), json.loads(str(bn["materials"])), center_ref)
    xc = nodes[quads].mean(1)[:, ax]; f = (xc - xc.min()) / (xc.max() - xc.min())
    Dq = {i: (1 - f[i]) * DN[int(sdN[i])] + f[i] * DN1[int(sdN1[i])] for i in range(len(quads))}
    Gq = {i: (1 - f[i]) * GN[int(sdN[i])] + f[i] * GN1[int(sdN1[i])] for i in range(len(quads))}
    subdom = np.arange(len(quads))
    k22_e = compute_k22(nodes[quads].mean(1), e2s, e3s, quads) if k22_mode == "general" \
        else _seg_k22(nodes, quads, e2s, e3s, cross, k22_mode)[0]
    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment(nodes, quads, subdom, e1s, e2s, e3s, Dq, Gq, k22_e, cross)
    Dhh, Dhe, Dhl, Dll, Dle = map(np.asarray, (Dhh, Dhe, Dhl, Dll, Dle))

    kb = None if k22_mode != "general" else "general"
    resL = solve_boundary_bundle(b, "L", center_ref, shear, k22=kb)      # true @ x_min
    resR = solve_boundary_bundle(bn, "L", center_ref, shear, k22=kb)     # true @ x_max (== seg N R)
    n2sL = np.asarray(b["L_node2seg"])
    X6L = np.asarray(bn["L_x"]); X5R = np.asarray(b["R_x"])              # coincident rings
    dmap = np.linalg.norm(X6L[:, None, :] - X5R[None, :, :], axis=2); jm = dmap.argmin(axis=1)
    if dmap[np.arange(len(jm)), jm].max() > 1e-6:
        raise ValueError("seg N R and seg N+1 L rings are not geometrically coincident")
    n2sR = np.asarray(b["R_node2seg"])[jm]

    def scatter(key):
        bd, bv = [], []
        for res, n2s in [(resL, n2sL), (resR, n2sR)]:
            Vv = res[key].reshape(-1, 5, 4)
            for i, sn in enumerate(n2s):
                for c in range(5):
                    bd.append(5 * int(sn) + c); bv.append(Vv[i, c, :])
        return np.array(bd), np.array(bv, float)

    bd0, bv0 = scatter("V0"); V0 = dirichlet_solve(Dhh, -Dhe, bd0, bv0)
    L = float(xc.max() - xc.min()); EB = (np.asarray(Dee) + V0.T @ Dhe) / L
    C, Psi = build_C_Psi_segment(nodes, quads, cross); Dc = C.T
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle), jnp.array(Psi), jnp.array(Dc))
    bd1, bv1 = scatter("V1"); V1 = dirichlet_solve(Dhh, np.asarray(bb), bd1, bv1)
    S, *_ = finalize_v1_and_compute_deff(
        jnp.array(V1), jnp.array(V0), jnp.array(EB),
        jnp.array(np.asarray(V0DllV0) / L), jnp.array(np.asarray(DhlV0) / L),
        jnp.array(np.asarray(DhlTV0Dle) / L), jnp.array(Psi), jnp.array(Dc))
    C6 = np.asarray(S); C6 = 0.5 * (C6 + C6.T)
    out = {"C6": C6, "EB": EB, "L": L, "origin": float(nodes[:, ax].mean())}
    if verbose:
        print("layup-tapered Timo 6x6 diag [EA,GA2,GA3,GJ,EI2,EI3]:",
              np.array2string(np.diag(C6), precision=4))
    return out


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
