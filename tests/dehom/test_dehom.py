"""
Dehomogenization tests — two-step MSG-TW strain recovery.

Validation rests on two exact MSG identities, so these catch sign/index bugs in
the strain operators and the recovery wiring:

* Step 1 (shell strain recovery): the recovered EB shell-strain field stores
  exactly the beam strain energy,  0.5 INT eps^T ABD eps ds == 0.5 st_m^T EB st_m.
* Step 2 (plate dehomogenization): the through-thickness 3D stress integrates
  back to the applied stress resultants,  INT sigma dz == (ABD @ shell_strain).
"""
import os
import sys
# this file lives in tests/dehom/ ; fe_jax is under <repo>/opensg_jax, data in tests/data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "opensg_jax"))
import numpy as np
import jax.numpy as jnp
import pytest

from fe_jax import (
    solve_tw_from_yaml, hermite_strain_operators, recover_shell_strains,
    dehomogenize, compute_ABD_matrix, plate_dehom_strain, stress_at_points,
)
from fe_jax.msg_dehom import _recov

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
YAML0 = os.path.join(DATA_DIR, "1Dshell_0.yaml")


@pytest.fixture(scope="module")
def bundle():
    if not os.path.exists(YAML0):
        pytest.skip(f"missing {YAML0}")
    return solve_tw_from_yaml(YAML0)


def _eb_shell_energy(bundle, st_m):
    """0.5 INT eps_EB^T ABD eps_EB ds for the pure-EB recovered shell strain."""
    V0 = np.asarray(bundle["V0"]); rc = np.asarray(bundle["red_cells"])
    corners = np.asarray(bundle["corners"]); ABD = np.asarray(bundle["ABD_elems"])
    k22 = np.asarray(bundle["k22"]); L = np.asarray(bundle["L"])
    xd2 = np.asarray(bundle["xd2"]); xd3 = np.asarray(bundle["xd3"])
    xi_q = bundle["xi_q"]; W_q = np.asarray(bundle["W_q"])
    w = V0 @ st_m
    st_m_j = jnp.array(st_m)
    U = 0.0
    for e in range(rc.shape[0]):
        c0, c1 = int(rc[e, 0]), int(rc[e, 1])
        g = np.r_[c0 * 6:c0 * 6 + 6, c1 * 6:c1 * 6 + 6]
        eps_h, eps_l, eps_e, _ = hermite_strain_operators(
            jnp.array(corners[c0]), jnp.array(corners[c1]), float(k22[e]),
            float(L[e]), float(xd2[e]), float(xd3[e]), xi_q)
        ss = np.asarray(eps_h(jnp.array(w[g])) + eps_e(st_m_j))  # (Q,6)
        for q in range(ss.shape[0]):
            U += 0.5 * ss[q] @ ABD[e] @ ss[q] * float(L[e]) * float(W_q[q])
    return U


@pytest.mark.parametrize("comp,name", [(0, "extension"), (1, "twist"),
                                        (2, "bending2"), (3, "bending3")])
def test_step1_eb_energy_identity(bundle, comp, name):
    """Recovered EB shell strain stores exactly the beam EB strain energy."""
    st_m = np.zeros(4); st_m[comp] = 1.0
    U = _eb_shell_energy(bundle, st_m)
    U_beam = 0.5 * float(bundle["EB"][comp, comp])
    # Identity is exact for an exactly-constrained V0; residual here is the KKT
    # constraint-penalty (1e8) level (largest for the high-warping bending mode).
    assert abs(U - U_beam) / abs(U_beam) < 1e-3, \
        f"{name}: shell energy {U:.6e} vs beam {U_beam:.6e}"


def test_step1_recover_runs(bundle):
    """Full Timoshenko shell-strain recovery returns finite strains."""
    FF = np.array([1.0e5, 2.0e3, 1.5e3, 5.0e2, 8.0e3, 7.0e3])
    out = recover_shell_strains(bundle, FF)
    assert out["shell_strain"].shape[2] == 6
    assert np.all(np.isfinite(out["shell_strain"]))
    assert out["shell_strain_elem"].shape == (out["shell_strain"].shape[0], 6)


def _simpson_resultants(z, Sig, n_per):
    """Per-sub-element Simpson integration of [N11,N22,N12,M11,M22,M12]."""
    z = z.reshape(-1, n_per); Sig = Sig.reshape(-1, n_per, 6)
    N = np.zeros(6)
    for el in range(z.shape[0]):
        he = z[el, -1] - z[el, 0]
        wq = np.array([1.0, 4.0, 1.0]) * he / 6.0          # Simpson (n_per=3)
        s11 = Sig[el, :, 0]; s22 = Sig[el, :, 1]; s12 = Sig[el, :, 5]
        zc = z[el]
        N[0] += wq @ s11;        N[1] += wq @ s22;        N[2] += wq @ s12
        N[3] += wq @ (s11 * zc); N[4] += wq @ (s22 * zc); N[5] += wq @ (s12 * zc)
    return N


def test_step2_plate_resultant_equilibrium(bundle):
    """3D stress integrates back to the ABD shell-force resultants."""
    info = next(iter(bundle["layup_db"].values()))
    ln = next(iter(bundle["layup_db"]))
    ABD, _, warp = compute_ABD_matrix(
        info["thick"], info["angles"], info["mat_names"],
        bundle["material_db"], return_warping=True)
    rng = np.random.default_rng(0)
    ss = rng.standard_normal(6)
    z, Gam, Sig = plate_dehom_strain(warp, ss, n_eval_per_elem=3)
    N = _simpson_resultants(z, Sig, 3)
    N_ref = ABD @ ss
    assert np.allclose(N, N_ref, rtol=1e-7, atol=1e-6 * np.abs(N_ref).max()), \
        f"resultants {N} vs ABD@ss {N_ref}"


def test_full_pipeline(bundle):
    """End-to-end two-step dehomogenization runs and is finite."""
    FF = np.array([1.0e5, 0.0, 0.0, 0.0, 1.0e4, 0.0])
    out = dehomogenize(YAML0, FF, bundle=bundle)
    assert len(out["elem"]) == out["shell_strain"].shape[0]
    for ed in out["elem"]:
        assert ed["strain_3d"].shape[1] == 6 and ed["stress_3d"].shape[1] == 6
        assert np.all(np.isfinite(ed["strain_3d"]))
        assert np.all(np.isfinite(ed["stress_3d"]))


# ----------------------------------------------------------------------------
# Corrected Timoshenko recovery chain + prescribed-strain / end-node / order-3
# ----------------------------------------------------------------------------

def test_recov_matrix_corrected():
    """R1 = recov([1,0,0,0,0,0]) is the corrected fixed operator (only two 1s)."""
    R1 = _recov(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    exp = np.zeros((6, 6)); exp[4, 2] = 1.0; exp[5, 1] = -1.0
    assert np.array_equal(R1, exp), R1


def test_step1_input_validation(bundle):
    """recover_shell_strains needs exactly one of force / strain."""
    with pytest.raises(ValueError):
        recover_shell_strains(bundle)
    with pytest.raises(ValueError):
        recover_shell_strains(bundle, beam_force_vabs=np.ones(6), beam_strain=np.ones(6))


def test_step1_prescribed_strain_matches_force(bundle):
    """beam_strain=st reproduces beam_force=Timo@st (compliance round-trip)."""
    st = np.array([3e-3, 1e-3, 8e-4, 2e-3, 1.5e-3, 1.2e-3])
    FF = np.asarray(bundle["Timo"]) @ st
    a = recover_shell_strains(bundle, beam_force_vabs=FF)
    b = recover_shell_strains(bundle, beam_strain=st)
    assert np.allclose(a["macro"], st, rtol=1e-8, atol=1e-12)
    assert np.allclose(a["macro"], b["macro"], rtol=1e-8, atol=1e-12)
    assert np.allclose(a["shell_strain"], b["shell_strain"], rtol=1e-8, atol=1e-12)


def test_step1_end_node_recovery(bundle):
    """xi_eval=[0,1] evaluates the shell strain at the element end nodes."""
    out = recover_shell_strains(bundle, beam_strain=np.full(6, 0.01),
                                xi_eval=[0.0, 1.0])
    rc = np.asarray(bundle["red_cells"]); corners = np.asarray(bundle["corners"])
    assert out["shell_strain"].shape == (rc.shape[0], 2, 6)
    assert np.all(np.isfinite(out["shell_strain"]))
    # the arc coordinates at xi=0,1 are exactly the two end-node coordinates
    assert np.allclose(out["x2"][:, 0], corners[rc[:, 0], 0])
    assert np.allclose(out["x3"][:, 1], corners[rc[:, 1], 1])


def test_step2_cubic_matches_quadratic(bundle):
    """4-node (cubic) plate element == 3-node (quadratic) for uniform plies."""
    info = next(iter(bundle["layup_db"].values()))
    args = (info["thick"], info["angles"], info["mat_names"], bundle["material_db"])
    A2, _, w2 = compute_ABD_matrix(*args, return_warping=True, elem_order=2)
    A3, _, w3 = compute_ABD_matrix(*args, return_warping=True, elem_order=3)
    assert np.max(np.abs(A2 - A3)) / np.max(np.abs(A2)) < 1e-5
    ss = np.array([1e-3, -2e-4, 5e-5, 1e-2, 2e-3, 1e-3])   # realistic shell strain
    z2, _, S2 = plate_dehom_strain(w2, ss, 3)
    z3, _, S3 = plate_dehom_strain(w3, ss, 3)
    s2, s3 = S2[np.argmin(z2)], S3[np.argmin(z3)]
    assert np.max(np.abs(s2 - s3)) / np.max(np.abs(s2)) < 1e-5


def test_dehomogenize_elem_order(bundle):
    """dehomogenize(elem_order=3) matches the default for uniform plies."""
    FF = np.array([1.0e5, 0.0, 0.0, 0.0, 1.0e4, 0.0])
    o2 = dehomogenize(YAML0, FF, bundle=bundle, elem_order=2)
    o3 = dehomogenize(YAML0, FF, bundle=bundle, elem_order=3)
    for e2, e3 in zip(o2["elem"], o3["elem"]):
        s2 = e2["stress_3d"][np.argmin(e2["z"])]
        s3 = e3["stress_3d"][np.argmin(e3["z"])]
        assert np.allclose(s2, s3, rtol=1e-5, atol=1.0)


def test_step2_n_per_layer_exact_and_covers_thickness(bundle):
    """2 plate sub-elements per layer is exact-equivalent to 1 (uniform plies),
    doubles the through-thickness samples, and spans the full laminate depth."""
    info = next(iter(bundle["layup_db"].values()))
    args = (info["thick"], info["angles"], info["mat_names"], bundle["material_db"])
    _, _, w1 = compute_ABD_matrix(*args, n_per_layer=1, return_warping=True)
    _, _, w2 = compute_ABD_matrix(*args, n_per_layer=2, return_warping=True)
    ss = np.array([1e-3, -2e-4, 5e-5, 1e-2, 2e-3, 1e-3])
    z1, _, S1 = plate_dehom_strain(w1, ss, 3)
    z2, _, S2 = plate_dehom_strain(w2, ss, 3)
    assert len(z2) == 2 * len(z1)                              # twice the samples
    h = float(sum(info["thick"]))
    assert abs(z2.min()) < 1e-12 and abs(z2.max() - h) < 1e-9  # full thickness
    # same physical stress profile: compare at the shared bottom face
    assert np.max(np.abs(S2[np.argmin(z2)] - S1[np.argmin(z1)])) \
        / np.max(np.abs(S1)) < 1e-6


def test_stress_at_points_n_per_layer_invariant(bundle):
    """stress_at_points is invariant to n_per_layer for uniform plies."""
    corners = np.asarray(bundle["corners"]); cen = corners.mean(axis=0)
    nd = corners[12]; p = nd + 0.03 * (cen - nd) / np.linalg.norm(cen - nd)
    FF = np.array([1.0e5, 5.0e4, 5.0e4, 5.0e4, 1.0e5, 1.0e5])
    s1 = stress_at_points(bundle, [p], beam_force_vabs=FF, n_per_layer=1)["stress"]
    s2 = stress_at_points(bundle, [p], beam_force_vabs=FF, n_per_layer=2)["stress"]
    assert np.max(np.abs(s2 - s1)) / np.max(np.abs(s1)) < 1e-6


# ----------------------------------------------------------------------------
# Point evaluation: stress at arbitrary cross-section coordinates
# ----------------------------------------------------------------------------

def test_stress_at_points_on_reference(bundle):
    """Reference (1D-mesh) nodes project to depth 0 and give finite stress."""
    corners = np.asarray(bundle["corners"])
    pts = corners[[0, 7, 15, 25]]
    FF = np.array([1.0e5, 5.0e4, 5.0e4, 5.0e4, 1.0e5, 1.0e5])
    out = stress_at_points(bundle, pts, beam_force_vabs=FF, frame="global")
    assert out["stress"].shape == (4, 6)
    assert np.all(np.isfinite(out["stress"]))
    assert np.allclose(out["depth"], 0.0, atol=1e-9)     # exactly on the OML


def test_stress_at_points_free_surface(bundle):
    """At the OML (z=0) the LOCAL transverse stresses sigma_33/13/23 ~ 0."""
    corners = np.asarray(bundle["corners"])
    out = stress_at_points(bundle, corners[[3, 9, 18]],
                           beam_strain=np.full(6, 1e-3), frame="local")
    sc = np.max(np.abs(out["stress"]))
    assert np.all(np.abs(out["stress"][:, [2, 3, 4]]) < 1e-3 * sc)   # S33,S23,S13


def test_stress_at_points_interior(bundle):
    """A point inside the wall projects to 0 < depth <= thickness, finite stress."""
    corners = np.asarray(bundle["corners"]); cen = corners.mean(axis=0)
    nd = corners[12]; inward = (cen - nd) / np.linalg.norm(cen - nd)
    p = nd + 0.03 * inward                               # ~0.03 into the 0.082 wall
    out = stress_at_points(bundle, [p], beam_strain=np.full(6, 1e-3))
    assert 0.0 < out["depth"][0] <= 0.0821
    assert np.all(np.isfinite(out["stress"]))


def test_stress_at_points_matches_pipeline(bundle):
    """stress_at_points global stress at an OML node == the laminate->global
    rotation of the plate-dehom OML stress for that element (self-consistency)."""
    corners = np.asarray(bundle["corners"]); rc = np.asarray(bundle["red_cells"])
    FF = np.array([1.0e5, 5.0e4, 5.0e4, 5.0e4, 1.0e5, 1.0e5])
    # a node that is the start corner of element e0 (xi=0 projection)
    e0 = 0; nd = int(rc[e0, 0])
    out = stress_at_points(bundle, corners[[nd]], beam_force_vabs=FF, frame="local")
    sh = recover_shell_strains(bundle, beam_force_vabs=FF, xi_eval=[0.0, 1.0])
    ss = sh["shell_strain"][e0, 0]                       # shell strain at that node
    info = bundle["layup_db"][bundle["layup_per_elem"][e0]]
    _, _, warp = compute_ABD_matrix(info["thick"], info["angles"],
        info["mat_names"], bundle["material_db"], return_warping=True)
    z, _, Sig = plate_dehom_strain(warp, ss, 3)
    assert np.allclose(out["stress"][0], Sig[np.argmin(z)], rtol=1e-6, atol=1.0)


if __name__ == "__main__":
    b = solve_tw_from_yaml(YAML0)
    for comp, nm in [(0, "extension"), (1, "twist"), (2, "bending2"), (3, "bending3")]:
        st_m = np.zeros(4); st_m[comp] = 1.0
        U = _eb_shell_energy(b, st_m); Ub = 0.5 * float(b["EB"][comp, comp])
        print(f"  {nm:10s}: shell U={U:.6e}  beam={Ub:.6e}  rel={abs(U-Ub)/abs(Ub):.2e}")
