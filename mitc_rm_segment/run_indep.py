"""run_indep.py -- solve the square segment with omega_3 as an INDEPENDENT 6th DOF
(segment_indep) and compare GA3/C33 to the FEniCS solid.  Penalty `pen` enforces the
finite drilling residual DR=0; sweep it to check convergence.

    python run_indep.py            # thin iso+m45 aR070, pen sweep
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..")); sys.path.insert(0, REPO)
BENCH = os.path.join(REPO, "examples", "data", "benchmark")
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]


def shell_solve_indep(tg, mesh_dir, res_dir, pen=None, pen_beta=0.1):
    import io, contextlib
    import jax.numpy as jnp
    from boundary_from_yaml import extract
    from segment_element import dirichlet_solve, compute_k22, compute_kg
    from segment_element_general import ring_general
    from segment_indep import assemble_segment_indep, build_C_Psi_segment6
    from solve_segment_jax import _material_by_section
    from opensg_jax.fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff

    npz = os.path.join(res_dir, "shell_%s.npz" % tg)
    with contextlib.redirect_stdout(io.StringIO()):
        extract(os.path.join(mesh_dir, "shell_%s.yaml" % tg), npz)
    b = np.load(npz, allow_pickle=True)
    ax = int(b["axis"]); cross = tuple(j for j in range(3) if j != ax)
    nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"]); sd = np.asarray(b["seg_subdom"])
    e1s, e2s, e3s = np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]), np.asarray(b["seg_e3"])
    D_by, G_by = _material_by_section(json.loads(str(b["sections"])), json.loads(str(b["materials"])), center_ref=True)
    cents = nodes[quads].mean(1)
    k22_e = compute_k22(cents, e2s, e3s, quads)
    kg_e = compute_kg(cents, e1s, e2s, e3s, quads)

    rings = {}
    for side in ("L", "R"):
        rx = np.asarray(b["%s_x" % side]); rc = np.asarray(b["%s_cells" % side])
        rs = np.asarray(b["%s_subdom" % side]); re3 = np.asarray(b["%s_e3" % side])
        kr = compute_k22(rx[rc].mean(1), np.asarray(b["%s_e2" % side]), re3, rc)
        C6r, V0r, V1r = ring_general(rx, rc, rs, re3, D_by, G_by, kr, ax, list(cross), shear="mitc4_both")
        rings[side] = dict(C6=C6r, V0=V0r, V1=V1r)

    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment_indep(
        nodes, quads, sd, e3s, D_by, G_by, k22_e, cross, ax, kg_e=kg_e, pen=pen, pen_beta=pen_beta)
    Dhh, Dhe, Dhl, Dll, Dle = map(np.asarray, (Dhh, Dhe, Dhl, Dll, Dle))

    def scatter(key):                                  # map 5-DOF ring -> first 5 of 6 DOFs; om3 free
        bd, bv = [], []
        for side in ("L", "R"):
            V = rings[side][key].reshape(-1, 5, 4)
            for i, sn in enumerate(np.asarray(b["%s_node2seg" % side])):
                for c in range(5):
                    bd.append(6 * int(sn) + c); bv.append(V[i, c, :])
        return np.array(bd), np.array(bv, float)

    bd0, bv0 = scatter("V0"); V0 = dirichlet_solve(Dhh, -Dhe, bd0, bv0)
    Lz = float(nodes[:, ax].max() - nodes[:, ax].min())
    EB = (np.asarray(Dee) + V0.T @ Dhe) / Lz
    C, Psi = build_C_Psi_segment6(nodes, quads, cross)
    Psi[3::6, 3] *= -1.0
    Dc = C.T
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle), jnp.array(Psi), jnp.array(Dc))
    bd1, bv1 = scatter("V1"); V1 = dirichlet_solve(Dhh, np.asarray(bb), bd1, bv1)
    S6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V1), jnp.array(V0), jnp.array(EB),
        jnp.array(np.asarray(V0DllV0) / Lz), jnp.array(np.asarray(DhlV0) / Lz),
        jnp.array(np.asarray(DhlTV0Dle) / Lz), jnp.array(Psi), jnp.array(Dc))
    return 0.5 * (np.asarray(S6) + np.asarray(S6).T)


def shell_solve_lagrange(tg, mesh_dir, res_dir, lam_space="elem", return_full=False):
    """ALL-6-DOF tapered segment homogenization: constrained (independent-omega3)
    element for BOTH the boundary rings and the segment interior, DR=0 imposed
    exactly via element-constant Lagrange multipliers.  The segment Dirichlet data
    includes the drilling omega_3 ring values.  return_full=True additionally
    returns the ring 6x6s and per-stage wall times."""
    import io, contextlib, time
    import jax.numpy as jnp
    from boundary_from_yaml import extract
    from segment_element import dirichlet_solve, compute_k22, compute_kg
    from segment_indep import assemble_segment_indep, assemble_constraint, build_C_Psi_segment6
    from solve_segment_jax import _material_by_section
    from opensg_jax.fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff
    from run_ring_indep import ring_indep

    t0 = time.perf_counter()
    npz = os.path.join(res_dir, "shell_%s.npz" % tg)
    with contextlib.redirect_stdout(io.StringIO()):
        extract(os.path.join(mesh_dir, "shell_%s.yaml" % tg), npz)
    b = np.load(npz, allow_pickle=True)
    ax = int(b["axis"]); cross = tuple(j for j in range(3) if j != ax)
    nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"]); sd = np.asarray(b["seg_subdom"])
    e1s, e2s, e3s = np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]), np.asarray(b["seg_e3"])
    D_by, G_by = _material_by_section(json.loads(str(b["sections"])), json.loads(str(b["materials"])), center_ref=True)
    cents = nodes[quads].mean(1)
    k22_e = compute_k22(cents, e2s, e3s, quads); kg_e = compute_kg(cents, e1s, e2s, e3s, quads)
    t_extract = time.perf_counter() - t0

    t0 = time.perf_counter()
    rings = {}
    for side in ("L", "R"):
        rx = np.asarray(b["%s_x" % side]); rc = np.asarray(b["%s_cells" % side])
        rs = np.asarray(b["%s_subdom" % side]); re3 = np.asarray(b["%s_e3" % side])
        kr = compute_k22(rx[rc].mean(1), np.asarray(b["%s_e2" % side]), re3, rc)
        C6r, V0r, V1r = ring_indep(rx, rc, rs, re3, D_by, G_by, kr, ax, list(cross),
                                   lam_space=lam_space, return_fields=True)
        rings[side] = dict(C6=C6r, V0=V0r, V1=V1r)
    t_rings = time.perf_counter() - t0

    t0 = time.perf_counter()
    # PRODUCTION shear scheme (segment): FULL integration.  Verified locking-free in
    # the worst case (prismatic flat-wall identity seg==ring to +-0.00% down to
    # t/R=2e-4, all meshes); canonical MITC tying breaks flat-walled/webbed sections
    # (square -29/-47%, webbed ellipse -17/+29%) by aliasing the algebraic drilling
    # content, and flux-only tying is numerically identical to full integration.
    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment_indep(
        nodes, quads, sd, e3s, D_by, G_by, k22_e, cross, ax, kg_e=kg_e, pen=0.0,
        shear="full")
    Gc, Gl, Ge = assemble_constraint(nodes, quads, sd, e3s, k22_e, cross, ax, kg_e=kg_e,
                                     lam_space=lam_space)
    Dhh, Dhe, Dhl, Dll, Dle = map(np.asarray, (Dhh, Dhe, Dhl, Dll, Dle))
    M = Dhh.shape[0]; P = Gc.shape[0]; naug = M + P

    Dhh_a = np.zeros((naug, naug)); Dhh_a[:M, :M] = Dhh; Dhh_a[:M, M:] = Gc.T; Dhh_a[M:, :M] = Gc
    Dhe_a = np.zeros((naug, 4)); Dhe_a[:M] = Dhe; Dhe_a[M:] = Ge
    Dhl_a = np.zeros((naug, naug)); Dhl_a[:M, :M] = Dhl; Dhl_a[M:, :M] = Gl
    Dll_a = np.zeros((naug, naug)); Dll_a[:M, :M] = Dll
    Dle_a = np.zeros((naug, 4)); Dle_a[:M] = Dle
    C, Psi = build_C_Psi_segment6(nodes, quads, cross); Psi[3::6, 3] *= -1.0
    Psi_a = np.zeros((naug, 4)); Psi_a[:M] = Psi
    Dc_a = np.zeros((naug, 4)); Dc_a[:M] = C.T

    def scatter(key):
        # 6-DOF ring fields: all six dofs (incl. the drilling omega_3) become
        # Dirichlet data for the segment
        bd, bv = [], []
        for side in ("L", "R"):
            V = rings[side][key].reshape(-1, 6, 4)
            for i, sn in enumerate(np.asarray(b["%s_node2seg" % side])):
                for c in range(6):
                    bd.append(6 * int(sn) + c); bv.append(V[i, c, :])
        return np.array(bd), np.array(bv, float)

    bd0, bv0 = scatter("V0"); V0 = dirichlet_solve(Dhh_a, -Dhe_a, bd0, bv0)
    Lz = float(nodes[:, ax].max() - nodes[:, ax].min())
    EB = (np.asarray(Dee) + V0.T @ Dhe_a) / Lz
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl_a), jnp.array(Dll_a), jnp.array(Dle_a), jnp.array(Psi_a), jnp.array(Dc_a))
    bd1, bv1 = scatter("V1"); V1 = dirichlet_solve(Dhh_a, np.asarray(bb), bd1, bv1)
    S6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V1), jnp.array(V0), jnp.array(EB),
        jnp.array(np.asarray(V0DllV0) / Lz), jnp.array(np.asarray(DhlV0) / Lz),
        jnp.array(np.asarray(DhlTV0Dle) / Lz), jnp.array(Psi_a), jnp.array(Dc_a))
    S6 = 0.5 * (np.asarray(S6) + np.asarray(S6).T)
    t_seg = time.perf_counter() - t0
    if return_full:
        return dict(S6=S6, C6L=rings["L"]["C6"], C6R=rings["R"]["C6"],
                    t_extract=t_extract, t_rings=t_rings, t_seg=t_seg)
    return S6


def show6(mat="m45", regime="thin", aR=0.7, pen_beta=0.1):
    """Full 6x6 Timoshenko (independent-omega3 shell) with per-entry %err vs the
    3-D solid taper, for the square section."""
    import taper_study as ts
    MESH = os.path.join(HERE, "out", "taper_square", "meshes")
    RES = os.path.join(HERE, "out", "taper_square", "results")
    b = np.load(os.path.join(BENCH, "taper_square_solid_%s.npz" % mat), allow_pickle=True)
    tg = ts.tag_of(regime, mat, aR)
    So = 0.5 * (b["%s_seg" % tg] + b["%s_seg" % tg].T)
    S6 = shell_solve_indep(tg, MESH, RES, pen_beta=pen_beta)
    thr = 1e-3 * abs(np.diag(So)).max()
    print("\n### %s SQUARE, TAPERED segment : INDEP-omega3 SHELL vs 3-D SOLID (pen_beta=%.2f) ###" % (tg, pen_beta))
    print("beam-strain order [ext, shear2, shear3, torsion, bend2, bend3] = [EA,GA2,GA3,GJ,EI2,EI3]\n")
    print("           " + "".join("%12s" % ("C%d%d" % (i + 1, i + 1)) for i in range(6)) + "   (diag, x1e9)")
    print("  solid:   " + "".join("%12.4f" % (So[i, i] / 1e9) for i in range(6)))
    print("  shell:   " + "".join("%12.4f" % (S6[i, i] / 1e9) for i in range(6)))
    print("  %err :   " + "".join("%11.1f%%" % (100 * (S6[i, i] - So[i, i]) / So[i, i]) for i in range(6)))
    print("\nFULL 6x6  %err on every |C_ij| > 0.1%% of max diag  (blank = both ~0):")
    print("      " + "".join("%9s" % ("C%d%d" % (j + 1, j + 1) if False else "j=%d" % (j + 1)) for j in range(6)))
    for i in range(6):
        row = "  i=%d " % (i + 1)
        for j in range(6):
            if abs(So[i, j]) > thr or abs(S6[i, j]) > thr:
                e = 100 * (S6[i, j] - So[i, j]) / So[i, j] if So[i, j] else float("nan")
                row += "%8.1f%%" % e
            else:
                row += "%9s" % "."
        print(row)
    print("\n  (raw shell 6x6, x1e9)")
    for i in range(6):
        print("   " + "".join("%12.4f" % (S6[i, j] / 1e9) for j in range(6)))


if __name__ == "__main__":
    import sys
    show6(sys.argv[1] if len(sys.argv) > 1 else "m45")
