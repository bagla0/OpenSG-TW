"""cyl_buckling_bench.py -- isotropic cylinder axial buckling, 3 ways:

  (1) ANALYTICAL   N_cr = E t^2 / (R sqrt(3(1-nu^2)))     [Timoshenko-Gere; Brush & Almroth]
  (2) JAX-FEA      uniform axial N imposed  ->  MITC4 shell buckling  (shell_buckling.solve_buckling)
  (3) RM-OpenSG    N from the MSG two-step chain: RM ring homogenization + beam dehom recovers the wall
                   membrane resultant N = A eps + B kappa, fed into the SAME buckling solver.

Both FE columns use the SS3 classical benchmark BC.  The point of (3): the cheap cross-section
homogenization + dehomogenization must reproduce the direct shell-FEA buckling load AND mode shapes.
Writes cyl_bench.json (loads/ratios/timing) and cyl_bench.npz (nodes/quads + mode fields for figures)."""
import os, sys, time, json
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__))                    # examples/buckling (this folder)
ROOT = os.path.abspath(os.path.join(BUCK, "..", ".."))              # OpenSG-TW repo root
XSEC = os.path.join(ROOT, "examples", "TW-paper", "xsec_paper")     # dehom_rm, emit_abd
sys.path.insert(0, ROOT); sys.path.insert(0, XSEC); sys.path.insert(0, BUCK)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax; jax.config.update("jax_enable_x64", True)
import shell_buckling as sb                                          # local (this folder) = core mirror

E, nu, R, t, L = 200e9, 0.3, 1.0, 0.02, 2.0
NC, NL, NMODES = 160, 80, 10
OUT = os.path.join(BUCK, "data")
Ncl = E * t ** 2 / (R * np.sqrt(3 * (1 - nu ** 2)))


# ---------- cylinder 3-D shell mesh (SS3) ----------
def cyl_mesh(nc, nl):
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    xs = np.linspace(0, L, nl + 1)
    nodes = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])] for i in range(nl + 1) for j in range(nc)])
    idx = lambda i, j: i * nc + (j % nc)
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                      for i in range(nl) for j in range(nc)])
    fixed = []
    for j in range(nc):
        r0, rL = idx(0, j), idx(nl, j)
        fixed += [6 * r0 + 1, 6 * r0 + 2, 6 * r0 + 0, 6 * rL + 1, 6 * rL + 2]   # SS3 both ends
    return nodes, quads, np.unique(fixed), th


# ---------- isotropic-cylinder RING 1-D shell yaml ----------
def write_ring_yaml(path, N=NC):
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    G = E / (2 * (1 + nu))
    pts = np.array([[R * np.cos(a), R * np.sin(a), 0.0] for a in th])
    L_ = ["nodes:"]
    for p in pts:
        L_.append("- [%.10f %.10f 0.00000000]" % (p[0], p[1]))
    L_.append("elements:")
    for i in range(N):
        L_.append("- [%d %d]" % (i + 1, (i + 1) % N + 1))          # closed loop 1..N -> 1
    L_.append("elementOrientations:")                               # per elem [e1, e2(tangent), e3=e1xe2]
    e1 = np.array([0.0, 0.0, 1.0])                                  # beam axis (z); cross-section in x-y
    for i in range(N):
        e2 = pts[(i + 1) % N] - pts[i]; e2 = e2 / (np.linalg.norm(e2) + 1e-30)
        e3 = np.cross(e1, e2)
        L_.append("- [%.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f]"
                  % (e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], e3[0], e3[1], e3[2]))
    L_ += ["sets:", "  element:", "  - name: layup_0", "    labels:"]
    L_ += ["    - %d" % (i + 1) for i in range(N)]
    L_ += ["sections:", "- type: shell", "  elementSet: layup_0", "  layup:",
           "  - - iso", "    - %.10f" % t, "    - 0.0"]
    L_ += ["materials:", "- name: iso", "  density: 1000.0", "  elastic:",
           "    E:", "    - %.6e" % E, "    - %.6e" % E, "    - %.6e" % E,
           "    G:", "    - %.6e" % G, "    - %.6e" % G, "    - %.6e" % G,
           "    nu:", "    - %.6f" % nu, "    - %.6f" % nu, "    - %.6f" % nu]
    L_ += ["reference: center"]
    open(path, "w").write("\n".join(L_) + "\n")


# ---------- RM-OpenSG: membrane N field from dehom ----------
def rm_opensg_N(ring_yaml, F_axial=-1.0):
    """Homogenize the ring, apply unit axial beam force, recover per-circ wall membrane N=[Nxx,Nyy,Nxy]."""
    import dehom_rm
    from emit_abd import load_station_abd
    t0 = time.time()
    B = dehom_rm.build_rm_bundle(ring_yaml)                       # RM homogenization
    t_homo = time.time() - t0
    C6 = np.asarray(B["Timo"])
    # ABD (mid-ref) from the compulsory emitted yaml -> used for BOTH K and N (fully pipeline-derived)
    ay = os.path.join(os.path.dirname(ring_yaml), "abd", os.path.splitext(os.path.basename(ring_yaml))[0] + "_abd.yaml")
    ABD6, Gs2, _thk = load_station_abd(ay)["by_name"]["layup_0"]     # (ABD 6x6, Gs 2x2, thickness)
    ABD6, Gs2 = np.asarray(ABD6), np.asarray(Gs2)
    A, Bm = ABD6[:3, :3], ABD6[:3, 3:]
    t1 = time.time()
    st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=[F_axial, 0, 0, 0, 0, 0])
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
    # N at each ring element (centered), tagged by its circumferential angle
    ne = len(rc); ang = np.zeros(ne); Nring = np.zeros((ne, 3))
    for e in range(ne):
        s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        Nring[e] = A @ s6[:3] + Bm @ s6[3:6]                      # [N11 axial, N22 hoop, N12]
        mid = 0.5 * (corners[int(rc[e, 0])] + corners[int(rc[e, 1])])
        ang[e] = np.arctan2(mid[1], mid[0])
    t_dehom = time.time() - t1
    return dict(C6=C6, ABD6=ABD6, Gs2=Gs2, ang=ang, Nring=Nring, t_homo=t_homo, t_dehom=t_dehom)


def N_field_for_mesh(th_mesh, ang, Nring):
    """Map ring N (by angle) onto the nc circumferential element positions of the 3-D mesh."""
    dth = th_mesh[1] - th_mesh[0]
    thmid = th_mesh + 0.5 * dth
    Ncirc = np.zeros((len(th_mesh), 3))
    for j, a in enumerate(thmid):
        k = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - a))))))   # nearest ring element by angle
        Ncirc[j] = Nring[k]
    return Ncirc


def run_buckling(nodes, quads, fixed, ABD, Gs, Nvec_e, nmodes=NMODES):
    ne = len(quads)
    ABD_e = np.repeat(np.asarray(ABD)[None], ne, 0)
    Gs_e = np.repeat(np.asarray(Gs)[None], ne, 0)
    t0 = time.time()
    loads, modes = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec_e, fixed, n_modes=nmodes)
    return loads, modes, time.time() - t0


if __name__ == "__main__":
    nodes, quads, fixed, th = cyl_mesh(NC, NL)
    ne = len(quads)
    print("cylinder mesh: %d nodes, %d quads, %d dof" % (len(nodes), ne, 6 * len(nodes)))

    # (2) JAX-FEA -- uniform imposed axial N = -1  (cached: 160s, skip on RM-only debug)
    ABD_iso, Gs_iso = sb._iso_ABD(E, nu, t)
    fea_cache = os.path.join(OUT, "cyl_fea_cache.npz")
    if os.path.exists(fea_cache) and "--refea" not in sys.argv:
        z = np.load(fea_cache); loads_fea, modes_fea, tf = z["loads"], z["modes"], float(z["tf"])
        print("JAX-FEA   : [cached]")
    else:
        Nfea = np.repeat(np.array([-1.0, 0.0, 0.0])[None], ne, 0)
        loads_fea, modes_fea, tf = run_buckling(nodes, quads, fixed, ABD_iso, Gs_iso, Nfea)
        np.savez(fea_cache, loads=loads_fea, modes=modes_fea, tf=tf)
    print("JAX-FEA   : N_cr=%.4e  ratio=%.4f  (%.1fs)  first: %s"
          % (loads_fea[0], loads_fea[0] / Ncl, tf, np.array2string(loads_fea[:5], precision=3)))

    # (3) RM-OpenSG -- N from homogenization + dehom
    ring_yaml = os.path.join(OUT, "iso_cyl_ring.yaml")
    write_ring_yaml(ring_yaml, NC)
    rm = rm_opensg_N(ring_yaml, F_axial=-1.0)
    print("  homogenized Timo: EA=%.4e (exact 2piR t E=%.4e); ABD vs iso A11 rm=%.4e iso=%.4e"
          % (rm["C6"][0, 0], 2 * np.pi * R * t * E, rm["ABD6"][0, 0], ABD_iso[0, 0]))
    print("  dehom N: N11(axial) mean=%.4e std=%.2e ; N22(hoop) mean=%.2e ; N12 mean=%.2e"
          % (rm["Nring"][:, 0].mean(), rm["Nring"][:, 0].std(), rm["Nring"][:, 1].mean(), rm["Nring"][:, 2].mean()))
    Ncirc = N_field_for_mesh(th, rm["ang"], rm["Nring"])
    Nrm = np.array([Ncirc[j] for i in range(NL) for j in range(NC)])       # broadcast axially
    loads_rm, modes_rm, trm = run_buckling(nodes, quads, fixed, rm["ABD6"], rm["Gs2"], Nrm)
    # RM buckling load in physical N/m: lambda * |N11 imposed|
    Nmag = abs(rm["Nring"][:, 0].mean())
    Ncr_rm = loads_rm[0] * Nmag
    print("RM-OpenSG : N_cr=%.4e  ratio=%.4f  (homo %.1fs + dehom %.1fs + buckle %.1fs)  first(xNmag): %s"
          % (Ncr_rm, Ncr_rm / Ncl, rm["t_homo"], rm["t_dehom"], trm,
             np.array2string(loads_rm[:5] * Nmag, precision=3)))

    # mode correlation (MAC) between FEA and RM first mode (radial component)
    def radial(modes, m):
        ur = np.zeros(len(nodes))
        for n in range(len(nodes)):
            y, z = nodes[n, 1], nodes[n, 2]; rr = np.hypot(y, z) + 1e-30
            ur[n] = (modes[n, 1, m] * y + modes[n, 2, m] * z) / rr
        return ur
    u_f, u_r = radial(modes_fea, 0), radial(modes_rm, 0)
    mac = (u_f @ u_r) ** 2 / ((u_f @ u_f) * (u_r @ u_r) + 1e-30)
    print("MAC(FEA mode1, RM mode1) = %.4f" % mac)

    out = dict(analytical=Ncl, JAX_FEA=dict(Ncr=float(loads_fea[0]), modes=[float(x) for x in loads_fea[:NMODES]]),
               RM_OpenSG=dict(Ncr=float(Ncr_rm), Nmag=float(Nmag),
                              modes=[float(x * Nmag) for x in loads_rm[:len(loads_rm)]],
                              t_homo=rm["t_homo"], t_dehom=rm["t_dehom"]),
               t_fea=tf, t_rm_buckle=trm, mesh=[int(NC), int(NL)], MAC=float(mac),
               EA=float(rm["C6"][0, 0]))
    json.dump(out, open(os.path.join(OUT, "cyl_bench.json"), "w"), indent=2)
    np.savez(os.path.join(OUT, "cyl_bench.npz"), nodes=nodes, quads=quads,
             modes_fea=modes_fea, modes_rm=modes_rm, loads_fea=loads_fea, loads_rm=loads_rm,
             ang=rm["ang"], Nring=rm["Nring"])
    print("wrote cyl_bench.json + cyl_bench.npz")
