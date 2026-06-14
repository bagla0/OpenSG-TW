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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "opensg_jax"))
import numpy as np
import jax.numpy as jnp
import pytest

from fe_jax import (
    solve_tw_from_yaml, hermite_strain_operators, recover_shell_strains,
    dehomogenize, compute_ABD_matrix, plate_dehom_strain,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
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


if __name__ == "__main__":
    b = solve_tw_from_yaml(YAML0)
    for comp, nm in [(0, "extension"), (1, "twist"), (2, "bending2"), (3, "bending3")]:
        st_m = np.zeros(4); st_m[comp] = 1.0
        U = _eb_shell_energy(b, st_m); Ub = 0.5 * float(b["EB"][comp, comp])
        print(f"  {nm:10s}: shell U={U:.6e}  beam={Ub:.6e}  rel={abs(U-Ub)/abs(Ub):.2e}")
