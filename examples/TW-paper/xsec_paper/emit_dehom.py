"""emit_dehom.py -- OpenSG RM-shell dehomogenization OUTPUT EMITTER (VABS-compatible local fields).

A tool: given ONE 1-D shell SG cross-section and its beam internal load, run the RM two-step
dehomogenization once and write the recovered LOCAL 3-D fields at a contour x through-thickness sample
grid, in the same column layout VABS uses -- so a user can drop them straight into their own pipeline
(strength checks, plotting, comparison with VABS) WITHOUT re-running the dehomogenization.

    from emit_dehom import emit_cross_section
    emit_cross_section("iea_s10_shell.yaml", FF, "out/iea_s10",
                       contour_xy=oml, wall_thickness=th,          # OML contour + local laminate thickness
                       NT=16, bd_out="bd_driver.out", beam_node=11)

Files written (<prefix> = your out path):
    <prefix>.SM   y2 y3 c11 c12 c13 c22 c23 c33   local stress  [Pa]   (VABS material-frame column order)
    <prefix>.EM   y2 y3 e11 e12 e13 e22 e23 e33   local strain
    <prefix>.U    id y2 y3 u1 u2 u3               TOTAL local disp [m]  (warping + beam disp/rotation)
    <prefix>.npz  P[N,NT,2] stress[N,NT,6] strain[N,NT,6] disp[N,NT,3] th[N]   (fast NumPy reload)

Conventions: internal Voigt order is [11,22,33,23,13,12]; the .SM/.EM columns are reordered to the VABS
order [11,12,13,22,23,33].  Depth runs OML (t=0) -> IML (t=1); NT sets the through-thickness refinement.
The stress/strain are the two-step recovery Sigma = C(zeta) [B(zeta) V0 + Ge(zeta)] eps_bar; the .U is the
full local displacement (dehom warping plus the rigid beam translation/rotation, if a BeamDyn node given).
"""
import os
import numpy as np
import dehom_rm

VABS = [0, 5, 4, 1, 3, 2]    # internal Voigt [11,22,33,23,13,12] -> VABS column order [11,12,13,22,23,33]


def _beam_kinematics(bd_out, node):
    """rigid beam translation u_g + linearized rotation C at a BeamDyn output node (VABS .U convention)."""
    L = [l for l in open(bd_out).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith("Time"):
            h = l.split(); r = np.array([rr.split() for rr in L[i + 2:]], float)[-1]
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")]); RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]]); t1, t2, t3 = RD[2], -RD[1], RD[0]
            return u_g, np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]])
    raise ValueError("no BeamDyn header in " + bd_out)


def emit_cross_section(shell_yaml, beam_force_vabs, out_prefix, contour_xy, wall_thickness,
                       NT=16, ref=None, bd_out=None, beam_node=None, frame="material", bundle=None):
    """Run the RM dehom ONCE on this cross-section and write VABS-like .SM/.EM/.U/.npz.

    shell_yaml       : 1-D shell SG yaml (mid-ref).  beam_force_vabs : [F1,F2,F3,M1,M2,M3] (VABS order).
    contour_xy       : (N,2) OML contour in the (0,0) reference-axis frame.
    wall_thickness   : (N,) local total laminate thickness (m) at each contour point.
    NT               : through-thickness sample layers (refinement).  ref : center/oml (default = yaml).
    bd_out, beam_node: optional -> add the rigid beam disp/rotation so .U is the TOTAL local displacement.
    Returns dict(P[N,NT,2], stress[N,NT,6], strain[N,NT,6], disp[N,NT,3]).
    """
    oml = np.asarray(contour_xy, float); N = len(oml)
    th = np.asarray(wall_thickness, float)
    B = bundle if bundle is not None else dehom_rm.build_rm_bundle(shell_yaml, ref=ref)
    tg = np.gradient(oml, axis=0); tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-30)
    nrm = np.column_stack([tg[:, 1], -tg[:, 0]])
    cen = oml.mean(0); flip = ((cen - oml) * nrm).sum(1) < 0; nrm[flip] *= -1        # inward normal
    tt = np.linspace(0.0, 1.0, NT)
    P = oml[:, None, :] + tt[None, :, None] * (th[:, None, None] * nrm[:, None, :])  # (N,NT,2) OML->IML
    pts = P.reshape(-1, 2)
    res = dehom_rm.stress_at_points(B, pts, beam_force_vabs=beam_force_vabs, frame=frame, n_per_layer=4)
    S = np.asarray(res["stress"]); E = np.asarray(res["strain"])                     # (N*NT, 6) Voigt
    W = np.asarray(dehom_rm.disp_at_points(B, pts, beam_force_vabs=beam_force_vabs))
    if bd_out is not None and beam_node is not None:
        u_g, C = _beam_kinematics(bd_out, beam_node)
        r3 = np.column_stack([np.zeros(len(pts)), pts[:, 0], pts[:, 1]])
        U = u_g + (C @ (W + r3).T).T - r3
    else:
        U = W
    d = os.path.dirname(out_prefix)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savetxt(out_prefix + ".SM", np.column_stack([pts, S[:, VABS]]), fmt="%.6e",
               header="y2 y3 c11 c12 c13 c22 c23 c33  RM local stress [Pa] (VABS order); %dx%d grid" % (N, NT))
    np.savetxt(out_prefix + ".EM", np.column_stack([pts, E[:, VABS]]), fmt="%.6e",
               header="y2 y3 e11 e12 e13 e22 e23 e33  RM local strain")
    ids = np.arange(1, len(pts) + 1)
    np.savetxt(out_prefix + ".U", np.column_stack([ids, pts, U]),
               fmt=["%d", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e"], header="id y2 y3 u1 u2 u3  RM TOTAL local disp [m]")
    np.savez(out_prefix + ".npz", P=P, stress=S.reshape(N, NT, 6), strain=E.reshape(N, NT, 6),
             disp=U.reshape(N, NT, 3), th=th)
    return dict(P=P, stress=S.reshape(N, NT, 6), strain=E.reshape(N, NT, 6), disp=U.reshape(N, NT, 3))
