"""General operators END-TO-END (one parametrization): general ring SG for the
boundaries + general segment.  1) ring vs analytic; 2) prismatic identity;
3) circle_iso_taper vs solid."""
import os, sys, json, math
import numpy as np
sys.path.insert(0, r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code")
sys.path.insert(0, r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\mitc_rm_segment")
import jax.numpy as jnp
from segment_element import dirichlet_solve, compute_k22, build_C_Psi_segment
from segment_element_general import assemble_segment_general, ring_general
from solve_segment_jax import _material_by_section
from opensg_jax.fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff
S = r"C:\Users\bagla0\AppData\Local\Temp\claude\C--Users-bagla0\91cf4f05-ed42-47e2-974c-813d98a91247\scratchpad"
LBL = ["C11", "C22", "C33", "C44", "C55", "C66"]


def run(npz):
    b = np.load(npz, allow_pickle=True)
    ax = int(b["axis"]); cross = tuple(j for j in range(3) if j != ax)
    nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"]); sd = np.asarray(b["seg_subdom"])
    e1s, e2s, e3s = np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]), np.asarray(b["seg_e3"])
    sections = json.loads(str(b["sections"])); materials = json.loads(str(b["materials"]))
    D_by, G_by = _material_by_section(sections, materials, center_ref=True)
    k22_e = compute_k22(nodes[quads].mean(1), e2s, e3s, quads)

    rings = {}
    for side in ("L", "R"):
        rx = np.asarray(b["%s_x" % side]); rc = np.asarray(b["%s_cells" % side])
        rs = np.asarray(b["%s_subdom" % side]); re3 = np.asarray(b["%s_e3" % side])
        kr = compute_k22(rx[rc].mean(1), np.asarray(b["%s_e2" % side]), re3, rc)
        C6r, V0r, V1r = ring_general(rx, rc, rs, re3, D_by, G_by, kr, ax, list(cross))
        rings[side] = dict(C6=C6r, V0=V0r, V1=V1r)

    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment_general(
        nodes, quads, sd, e1s, e2s, e3s, D_by, G_by, k22_e, cross)
    Dhh, Dhe, Dhl, Dll, Dle = map(np.asarray, (Dhh, Dhe, Dhl, Dll, Dle))

    def scatter(key):
        bd, bv = [], []
        for side in ("L", "R"):
            V = rings[side][key].reshape(-1, 5, 4)
            for i, sn in enumerate(np.asarray(b["%s_node2seg" % side])):
                for c in range(5):
                    bd.append(5 * int(sn) + c); bv.append(V[i, c, :])
        return np.array(bd), np.array(bv, float)

    bd0, bv0 = scatter("V0"); V0 = dirichlet_solve(Dhh, -Dhe, bd0, bv0)
    L = float(nodes[:, ax].max() - nodes[:, ax].min())
    EB = (np.asarray(Dee) + V0.T @ Dhe) / L
    C, Psi = build_C_Psi_segment(nodes, quads, cross)
    Psi[3::5, 3] *= -1.0                     # general-op twist kernel: om1 = +1
    Dc = C.T
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle), jnp.array(Psi), jnp.array(Dc))
    bd1, bv1 = scatter("V1"); V1 = dirichlet_solve(Dhh, np.asarray(bb), bd1, bv1)
    S6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V1), jnp.array(V0), jnp.array(EB),
        jnp.array(np.asarray(V0DllV0) / L), jnp.array(np.asarray(DhlV0) / L),
        jnp.array(np.asarray(DhlTV0Dle) / L), jnp.array(Psi), jnp.array(Dc))
    S6 = np.asarray(S6); return 0.5 * (S6 + S6.T), rings


def ana(R, t=0.1, E=70e9, nu=0.3):
    G = E / (2 * (1 + nu)); A = 2 * math.pi * R * t; I = math.pi * R**3 * t
    return np.array([E * A, 0.5 * G * A, 0.5 * G * A, 2 * G * I, E * I, E * I])


print("=== PRISMATIC (circle_iso_prism, R=1): general ring + general segment ===")
C6p, rings = run(os.path.join(S, "shell_circle_iso_prism.npz"))
a = ana(1.0)
dL = np.diag(rings["L"]["C6"])
print("  %-5s %12s %10s | %12s %10s" % ("Cij", "ringGEN", "%err", "segGEN", "%err"))
for i in range(6):
    print("  %-5s %12.4e %+9.2f%% | %12.4e %+9.2f%%"
          % (LBL[i], dL[i], 100*(dL[i]-a[i])/a[i], C6p[i, i], 100*(C6p[i, i]-a[i])/a[i]))

print("\n=== TAPER (circle_iso_taper) vs SOLID ===")
C6t, ringsT = run(os.path.join(S, "shell_circle_iso_taper.npz"))
So = np.load(S + "/solid_circle_iso_taper_6x6.npy")
for i in range(6):
    print("  %-5s %10.4f %10.4f %+7.1f%%" % (LBL[i], C6t[i, i]/1e9, So[i, i]/1e9,
                                             100*(C6t[i, i]-So[i, i])/So[i, i]))
np.save(S + "/shell_circle_iso_taper_GENERAL2_6x6.npy", C6t)
