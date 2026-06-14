"""
MSG thin-walled (TW) dehomogenization — two-step strain recovery.

Given a solved cross-section (Timoshenko 6x6, plus the EB warping ``V0`` and the
shear warping ``V1``) and an applied beam force, recover the local strain field
in two steps:

**Step 1 — shell strain recovery** (this module, :func:`recover_shell_strains`).
Mirrors the OpenSG solid ``stress_recov.local_strain`` but with the MSG *shell*
strain operators (``eps_h`` / ``eps_l`` / ``eps_e`` from :mod:`msg_hermite`):

    shell_strain = eps_h(V0 @ st_m) + Ge @ st_m              (Euler-Bernoulli)
                 + eps_h(V1 @ st_cl1) + eps_l(V0 @ st_cl1)
                 + eps_l(V1 @ st_cl2)                         (Timoshenko)

``st_m`` is the macro EB strain from the compliance, and ``st_cl1`` / ``st_cl2``
are the Timoshenko derivative-recovery terms built from the ``recov`` chain
(identical to the solid).  The result is the 6 plate/shell strains
[eps11, eps22, gamma12, kappa11, kappa22, kappa12] at every cross-section
quadrature point.

**Step 2 — plate dehomogenization** (:func:`dehomogenize`, using
:func:`msg_materials.plate_dehom_strain`).  For each line element the 6 shell
strains drive the same MSG plate (through-thickness 1D SG) model that produced
the ABD matrix, recovering the pointwise 3D strain/stress across the thickness.

Force convention: VABS order ``[F1, F2, F3, M1, M2, M3]`` = [axial, shear-2,
shear-3, torsion, bending-2, bending-3].  (From BeamDyn reactions use
``FF = [rf[2], -rf[1], rf[0], rf[5], -rf[4], rf[3]]``.)
"""
import numpy as np
import jax.numpy as jnp

from .msg_hermite import hermite_strain_operators, solve_tw_from_yaml
from .msg_materials import compute_ABD_matrix, plate_dehom_strain

# EB<->shear coupling map (same Q as the solid recovery / finalize_v1)
_Q = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, -1.0], [1.0, 0.0]])


def _recov(st):
    """6x6 Timoshenko derivative-recovery matrix (matches solid ``utils.recov``)."""
    R = np.zeros((6, 6))
    R[0, 1], R[0, 2] = st[5], -st[4]
    R[1, 0], R[1, 2] = -st[5], st[3]
    R[2, 0], R[2, 1] = st[4], -st[3]
    R[3:6, 3:6] = R[0:3, 0:3]
    R[3, 1], R[3, 2] = st[2], -st[1]
    R[4, 0], R[4, 2] = -st[2], st[0]
    R[5, 0], R[5, 1] = st[1], -st[0]
    return R


def _macro_recovery(Ceff_srt, FF):
    """Macro EB strain ``st_m`` and the two Timoshenko derivative terms.

    Returns ``(st, st_m, st_cl1, st_cl2)``: ``st`` the full 6 beam strains,
    ``st_m`` the Q-corrected 4 EB strains, and ``st_cl1`` / ``st_cl2`` the
    Q-corrected first/second derivative recoveries (each length 4).
    """
    Comp = np.linalg.inv(Ceff_srt)
    st = Comp @ FF
    st_m = np.array([st[0], st[3], st[4], st[5]])

    F_1d = Ceff_srt @ st
    R1 = _recov(np.array([st[0] + 1.0, st[1], st[2], st[3], st[4], st[5]]))
    F1 = R1 @ F_1d
    stT1 = Comp @ F1
    st_cl1 = np.array([stT1[0], stT1[3], stT1[4], stT1[5]]); gamma1 = stT1[[1, 2]]

    R2 = _recov(stT1)
    F2 = R1 @ F1 + R2 @ F_1d
    stT2 = Comp @ F2
    st_cl2 = np.array([stT2[0], stT2[3], stT2[4], stT2[5]]); gamma2 = stT2[[1, 2]]

    R3 = _recov(stT2)
    F3 = 2.0 * R2 @ F1 + R3 @ F_1d + R1 @ F2
    stT3 = Comp @ F3; gamma3 = stT3[[1, 2]]

    st_m = st_m + _Q @ gamma1
    st_cl1 = st_cl1 + _Q @ gamma2
    st_cl2 = st_cl2 + _Q @ gamma3
    return st, st_m, st_cl1, st_cl2


def recover_shell_strains(bundle, beam_force_vabs):
    """Step 1 — recover the 6 shell strains at every cross-section quad point.

    Parameters
    ----------
    bundle : dict — from :func:`msg_hermite.solve_tw_from_yaml`
    beam_force_vabs : (6,) — applied beam force in VABS order

    Returns
    -------
    dict with:
      ``shell_strain`` (E, Q, 6) — [eps11, eps22, gamma12, kappa11, kappa22, kappa12]
      ``shell_strain_elem`` (E, 6) — quad-point-averaged per element
      ``x2``, ``x3`` (E, Q) — arc-point coordinates
      ``macro`` (6,) — beam strain ``st``;  ``st_m`` (4,) — EB macro strain
    """
    Ceff_srt = np.asarray(bundle["Timo"])
    V0 = np.asarray(bundle["V0"]); V1 = np.asarray(bundle["V1"])
    FF = np.asarray(beam_force_vabs, dtype=float)

    st, st_m, st_cl1, st_cl2 = _macro_recovery(Ceff_srt, FF)

    a1 = V0 @ st_m           # EB warping        (w_1)
    a2 = V1 @ st_cl1         # shear warping     (w1s_1)
    a3 = V1 @ st_cl2         # shear warping     (w1s_2)
    a4 = V0 @ st_cl1         # EB-of-derivative  (w_2)
    st_m_j = jnp.array(st_m)

    rc = np.asarray(bundle["red_cells"]); corners = np.asarray(bundle["corners"])
    k22 = np.asarray(bundle["k22"]); L = np.asarray(bundle["L"])
    xd2 = np.asarray(bundle["xd2"]); xd3 = np.asarray(bundle["xd3"])
    xi_q = bundle["xi_q"]; xin = np.asarray(xi_q)
    n_elem = rc.shape[0]; Q_pts = xin.shape[0]

    shell = np.zeros((n_elem, Q_pts, 6))
    x2 = np.zeros((n_elem, Q_pts)); x3 = np.zeros((n_elem, Q_pts))
    for e in range(n_elem):
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6]
        n0 = corners[c0]; n1 = corners[c1]
        eps_h, eps_l, eps_e, _ = hermite_strain_operators(
            jnp.array(n0), jnp.array(n1), float(k22[e]), float(L[e]),
            float(xd2[e]), float(xd3[e]), xi_q)
        s = (eps_h(jnp.array(a1[g])) + eps_e(st_m_j)
             + eps_h(jnp.array(a2[g])) + eps_l(jnp.array(a4[g]))
             + eps_l(jnp.array(a3[g])))
        shell[e] = np.asarray(s)
        x2[e] = (1.0 - xin) * n0[0] + xin * n1[0]
        x3[e] = (1.0 - xin) * n0[1] + xin * n1[1]

    return {"shell_strain": shell, "shell_strain_elem": shell.mean(axis=1),
            "x2": x2, "x3": x3, "macro": st, "st_m": st_m}


def dehomogenize(yaml_path, beam_force_vabs, n_eval_per_elem=3, bundle=None):
    """Full two-step dehomogenization for a YAML cross-section.

    Step 1 recovers the shell strains; step 2 runs the MSG plate model per
    element to recover the 3D strain/stress across the thickness (using the
    element-mean shell strain as the plate input).

    Parameters
    ----------
    yaml_path : str — OpenSG cross-section YAML
    beam_force_vabs : (6,) — applied beam force in VABS order
    n_eval_per_elem : int — through-thickness sample points per plate sub-element
    bundle : dict, optional — a precomputed :func:`solve_tw_from_yaml` bundle
             (avoids re-solving when dehomogenizing several load cases)

    Returns
    -------
    dict with the step-1 fields plus, for step 2:
      ``elem`` — list (length E) of per-element dicts
                 {``z``, ``strain_3d`` (n,6), ``stress_3d`` (n,6), ``layup``}
      ``bundle`` — the solution bundle (for reuse / inspection)
    """
    if bundle is None:
        bundle = solve_tw_from_yaml(yaml_path)
    shell = recover_shell_strains(bundle, beam_force_vabs)

    # plate warping cache per layup name
    warp_cache = {}
    for ln, info in bundle["layup_db"].items():
        _, _, warp = compute_ABD_matrix(
            info["thick"], info["angles"], info["mat_names"],
            bundle["material_db"], return_warping=True)
        warp_cache[ln] = warp

    ss_elem = shell["shell_strain_elem"]
    layups = bundle["layup_per_elem"]
    elem = []
    for e in range(len(layups)):
        ln = layups[e]
        z, Gam, Sig = plate_dehom_strain(warp_cache[ln], ss_elem[e], n_eval_per_elem)
        elem.append({"z": z, "strain_3d": Gam, "stress_3d": Sig, "layup": ln})

    out = dict(shell)
    out["elem"] = elem
    out["bundle"] = bundle
    return out
