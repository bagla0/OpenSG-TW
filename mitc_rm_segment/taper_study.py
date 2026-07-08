"""
taper_study.py -- incremental-taper CONVERGENCE study for the GENERAL RM
tapered-segment operators (segment_element_general), benchmarked against the
FEniCS 3-D solid at identical geometry/material/orientation.

    python taper_study.py gen               write all shell+solid meshes + e1/e2/e3 arrow PNGs
    python taper_study.py locking           shear-locking probe (thin prismatic, all MITC schemes)
    python taper_study.py shell [thin|thick] [mat]   run shell L-ring/R-ring/taper for the sweep
    python taper_study.py tables [thin|thick] [mat]  L/taper/R tables: solid | RM | %err

Sweep: circular tube, R_left = 1.0, R_right in TAPERS (1.0 = prismatic .. 0.7),
L = 2.0, mid-surface (CENTER reference) mesh, constant wall thickness:
    thin  t = 0.02  (t/R = 0.02)      thick t = 0.2  (t/R = 0.2)
Materials: iso (E=70 GPa, nu=0.3) and m45 (single-ply [-45], E1=37/E2=9/G=4 GPa).
Frames: shell e1=axial, e2=hoop tangent (CCW), e3=INWARD normal (reference);
solid EE1=fiber, N=INWARD through-thickness -- written identically so the
material orientation matches by construction (checked + plotted per case).
The solid runner (WSL) is run_solid_study.py on the same meshes.
"""
import os, sys, math, json
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(HERE, "out", "taper_study")
MESH = os.path.join(OUT, "meshes")
ORI = os.path.join(OUT, "orientation")
RES = os.path.join(OUT, "results")
for d in (MESH, ORI, RES):
    os.makedirs(d, exist_ok=True)

TAPERS = [1.0, 0.95, 0.9, 0.8, 0.7]
THICK = {"thin": 0.02, "thick": 0.2}
NC, NL, NR, L, R0 = 48, 10, 4, 2.0, 1.0
ISO = dict(name="iso", rho=1800.0, E=[70e9] * 3, G=[26.923e9] * 3, nu=[0.3] * 3)
ANI = dict(name="ani", rho=1800.0, E=[37e9, 9e9, 9e9], G=[4e9] * 3, nu=[0.3] * 3)
LBL = ["C11", "C22", "C33", "C44", "C55", "C66"]


def tag_of(regime, mat, aR):
    return "%s_%s_aR%03d" % (regime, mat, round(aR * 100))


# ------------------------------------------------------------------ mesh gen
def gen_case(regime, mat, aR, mesh_dir=None, nc=None, nl=None, nr=None):
    mesh_dir = mesh_dir or MESH
    os.makedirs(mesh_dir, exist_ok=True)
    nc = nc or NC; nl = nl or NL; nr = nr or NR       # circumferential / axial / through-thickness
    t = THICK[regime]
    matc = ISO if mat == "iso" else ANI
    layup = [["iso", t, 0.0]] if mat == "iso" else [["ani", t, -45.0]]   # single ply
    tg = tag_of(regime, mat, aR)
    Z = [L * i / nl for i in range(nl + 1)]

    def radius(z):
        return R0 + (aR * R0 - R0) * (z / L)

    dRdz = (aR * R0 - R0) / L                                 # taper rate (linear)

    def surf_frame(th):
        """SURFACE-FOLLOWING frame on the tapered wall (the reference frame):
        e1 = in-surface AXIAL tangent = the slanted generator line,
        e2 = hoop tangent (CCW),
        e3 = TRUE INWARD surface normal (perp to the tapered wall, not radial).
        For a prismatic tube (dRdz=0) this reduces to the axial/hoop/inward triad."""
        gen = np.array([dRdz * math.cos(th), dRdz * math.sin(th), 1.0])
        e1 = (gen / np.linalg.norm(gen)).tolist()
        e2 = [-math.sin(th), math.cos(th), 0.0]
        nrm = np.array([-math.cos(th), -math.sin(th), dRdz])   # inward, perp to surface
        e3 = (nrm / np.linalg.norm(nrm)).tolist()
        return e1, e2, e3

    # ---- shell (mid-surface) ----
    snodes, squads, soris = [], [], []
    for i, z in enumerate(Z):
        r = radius(z)
        for k in range(nc):
            th = 2 * math.pi * k / nc
            snodes.append([r * math.cos(th), r * math.sin(th), float(z)])
    for i in range(nl):
        for k in range(nc):
            k1 = (k + 1) % nc
            n0 = i * nc + k; n1 = i * nc + k1; n2 = (i + 1) * nc + k1; n3 = (i + 1) * nc + k
            squads.append([n0 + 1, n1 + 1, n2 + 1, n3 + 1])
            e1, e2, e3 = surf_frame(2 * math.pi * (k + 0.5) / nc)
            soris.append(e1 + e2 + e3)
    shell = {"nodes": snodes, "elements": squads,
             "sections": [{"elementSet": "wall", "layup": layup}],
             "sets": {"element": [{"name": "wall", "labels": list(range(1, len(squads) + 1))}]},
             "materials": [{"name": matc["name"], "density": matc["rho"],
                            "elastic": {"E": matc["E"], "G": matc["G"], "nu": matc["nu"]}}],
             "elementOrientations": soris}
    yaml.safe_dump(shell, open(os.path.join(mesh_dir, "shell_%s.yaml" % tg), "w"),
                   default_flow_style=None, sort_keys=False)

    # ---- solid (nr through-thickness layers about the mid-surface) ----
    def nid(i, m, k):
        return i * ((nr + 1) * nc) + m * nc + (k % nc) + 1
    dn = []
    for i, z in enumerate(Z):
        r = radius(z)
        for m in range(nr + 1):
            rr = r + (m / nr - 0.5) * t
            for k in range(nc):
                th = 2 * math.pi * k / nc
                dn.append("%.10f %.10f %.10f" % (rr * math.cos(th), rr * math.sin(th), z))
    hexes, hories = [], []
    for i in range(nl):
        for m in range(nr):
            for k in range(nc):
                k1 = (k + 1) % nc
                hexes.append(" ".join(str(x) for x in
                                      [nid(i, m, k), nid(i, m, k1), nid(i, m + 1, k1), nid(i, m + 1, k),
                                       nid(i + 1, m, k), nid(i + 1, m, k1), nid(i + 1, m + 1, k1), nid(i + 1, m + 1, k)]))
                a1, hoopl, nrm = surf_frame(2 * math.pi * (k + 0.5) / nc)
                a1 = np.array(a1); hoop = np.array(hoopl); e3 = np.array(nrm)
                if mat == "iso":
                    e1 = a1                                              # fiber = in-surface axial
                else:                                                    # fiber at -45 IN the surface
                    ca, sa = math.cos(math.radians(-45)), math.sin(math.radians(-45))
                    e1 = ca * a1 + sa * hoop
                e1 = e1 / np.linalg.norm(e1)
                e2 = np.cross(e3, e1)
                hories.append([float(v) for v in np.concatenate([e1, e2, e3])])
    mats = [{"name": matc["name"], "E": matc["E"], "G": matc["G"], "nu": matc["nu"], "rho": matc["rho"]}]
    solid = {"nodes": [[s] for s in dn], "elements": [[h] for h in hexes],
             "sets": {"element": [{"name": matc["name"], "labels": list(range(1, len(hexes) + 1))}]},
             "elementOrientations": hories, "materials": mats}
    yaml.safe_dump(solid, open(os.path.join(mesh_dir, "solid_%s.yaml" % tg), "w"),
                   default_flow_style=None, sort_keys=False)
    return tg


def _iso_cam(p):
    """Consistent SPANWISE ISOMETRIC camera for every taper render: parallel
    (orthographic) projection so the taper is shown without perspective
    foreshortening, beam axis z kept up, same view angle across all figures."""
    p.enable_parallel_projection()
    p.camera_position = [(6.0, -6.0, 4.3), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
    p.reset_camera()


def arrow_pngs(tg, mesh_dir=None, ori_dir=None):
    mesh_dir = mesh_dir or MESH; ori_dir = ori_dir or ORI
    os.makedirs(ori_dir, exist_ok=True)
    """SEPARATE e1(red)/e2(blue)/e3(black) arrow images for shell AND solid,
    e3 = inward normal (annotated with the measured e3.r_hat)."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    for kind in ("shell", "solid"):
        d = yaml.safe_load(open(os.path.join(mesh_dir, "%s_%s.yaml" % (kind, tg))))
        if kind == "shell":
            nodes = np.array([[float(v) for v in n] for n in d["nodes"]])
            elems = np.array([[int(v) - 1 for v in e] for e in d["elements"]])
        else:
            nodes = np.array([[float(v) for v in n[0].split()] for n in d["nodes"]])
            elems = np.array([[int(v) - 1 for v in e[0].split()] for e in d["elements"]])
        ori = np.array(d["elementOrientations"])
        cent = nodes[elems].mean(1)
        rhat = cent.copy(); rhat[:, 2] = 0.0
        rhat /= np.linalg.norm(rhat, axis=1)[:, None] + 1e-30
        e3dot = float(np.mean(np.sum(ori[:, 6:9] * rhat, axis=1)))
        step = max(1, len(cent) // 240)
        for vec, off, col in (("e1", 0, "red"), ("e2", 3, "blue"), ("e3", 6, "black")):
            p = pv.Plotter(off_screen=True, window_size=(950, 850))
            pd = pv.PolyData(cent[::step])
            pd["v"] = ori[::step, off:off + 3]
            p.add_mesh(pd.glyph(orient="v", scale=False, factor=0.18), color=col)
            p.add_mesh(pv.PolyData(nodes), color="#bbbbbb", point_size=2.0,
                       render_points_as_spheres=True)
            p.add_axes(line_width=3)
            _iso_cam(p)
            note = "  (e3.r_hat = %+.2f -> INWARD)" % e3dot if vec == "e3" else ""
            p.add_text("%s %s : %s%s" % (kind.upper(), vec, tg, note), font_size=11)
            p.screenshot(os.path.join(ori_dir, "%s_%s_%s.png" % (tg, kind, vec)))
            p.close()


def _load_mesh(tg, kind, mesh_dir):
    d = yaml.safe_load(open(os.path.join(mesh_dir, "%s_%s.yaml" % (kind, tg))))
    if kind == "shell":
        nodes = np.array([[float(v) for v in n] for n in d["nodes"]])
        elems = np.array([[int(v) - 1 for v in e] for e in d["elements"]])
    else:
        nodes = np.array([[float(v) for v in n[0].split()] for n in d["nodes"]])
        elems = np.array([[int(v) - 1 for v in e[0].split()] for e in d["elements"]])
    return nodes, elems, np.array(d["elementOrientations"])


def arrow_strip(tg, kind, mesh_dir=None, out_dir=None):
    """One HORIZONTAL image with e1(red) | e2(blue) | e3(black) side by side for
    a single body (kind='shell' or 'solid').  e3 is the inward normal (its mean
    e3.r_hat is annotated).  Returns the PNG path."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    mesh_dir = mesh_dir or MESH; out_dir = out_dir or ORI
    os.makedirs(out_dir, exist_ok=True)
    nodes, elems, ori = _load_mesh(tg, kind, mesh_dir)
    cent = nodes[elems].mean(1)
    rhat = cent.copy(); rhat[:, 2] = 0.0
    rhat /= np.linalg.norm(rhat, axis=1)[:, None] + 1e-30
    e3dot = float(np.mean(np.sum(ori[:, 6:9] * rhat, axis=1)))
    step = max(1, len(cent) // 220)
    p = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1650, 620), border=False)
    for c, (vec, off, col) in enumerate((("e1", 0, "red"), ("e2", 3, "blue"), ("e3", 6, "black"))):
        p.subplot(0, c)
        pd = pv.PolyData(cent[::step]); pd["v"] = ori[::step, off:off + 3]
        p.add_mesh(pd.glyph(orient="v", scale=False, factor=0.20), color=col)
        p.add_mesh(pv.PolyData(nodes), color="#cccccc", point_size=1.5, render_points_as_spheres=True)
        note = "  (e3.rhat=%+.2f, inward)" % e3dot if vec == "e3" else ""
        p.add_text("%s  %s%s" % (kind.upper(), vec, note), font_size=10)
        _iso_cam(p)
    fn = os.path.join(out_dir, "%s_%s_strip.png" % (tg, kind))
    p.screenshot(fn); p.close()
    return fn


def tapered_mesh_png(tg, kind, mesh_dir=None, out_dir=None):
    """3-D image of the TAPERED mesh (shell mid-surface quads, or solid hex body),
    one body per image.  Returns the PNG path."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    mesh_dir = mesh_dir or MESH; out_dir = out_dir or ORI
    os.makedirs(out_dir, exist_ok=True)
    nodes, elems, _ = _load_mesh(tg, kind, mesh_dir)
    if kind == "shell":
        faces = np.hstack([np.full((len(elems), 1), 4), elems]).ravel()
        grid = pv.PolyData(nodes, faces)
        col = "#4c78a8"
    else:
        cells = np.hstack([np.full((len(elems), 1), 8), elems]).ravel()
        ct = np.full(len(elems), pv.CellType.HEXAHEDRON, np.uint8)
        grid = pv.UnstructuredGrid(cells, ct, nodes)
        col = "#e07b39"
    p = pv.Plotter(off_screen=True, window_size=(760, 900))
    p.add_mesh(grid, color=col, show_edges=True, edge_color="#404040", line_width=0.6, opacity=1.0)
    p.add_text("%s tapered mesh : %s" % (kind.upper(), tg), font_size=11)
    p.add_axes(line_width=3)
    _iso_cam(p)
    fn = os.path.join(out_dir, "%s_%s_mesh.png" % (tg, kind))
    p.screenshot(fn); p.close()
    return fn


# ------------------------------------------------------------------ shell solve
def shell_solve(tg, shear="mitc4_both", mesh_dir=None, res_dir=None):
    mesh_dir = mesh_dir or MESH; res_dir = res_dir or RES
    os.makedirs(res_dir, exist_ok=True)
    import io, contextlib
    import jax.numpy as jnp
    from boundary_from_yaml import extract
    from segment_element import dirichlet_solve, compute_k22, compute_kg, build_C_Psi_segment
    from segment_element_general import assemble_segment_general, ring_general
    from solve_segment_jax import _material_by_section
    from opensg_jax.fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff

    npz = os.path.join(res_dir, "shell_%s.npz" % tg)
    with contextlib.redirect_stdout(io.StringIO()):
        extract(os.path.join(mesh_dir, "shell_%s.yaml" % tg), npz)
    b = np.load(npz, allow_pickle=True)
    ax = int(b["axis"]); cross = tuple(j for j in range(3) if j != ax)
    nodes = np.asarray(b["seg_x"]); quads = np.asarray(b["seg_cells"]); sd = np.asarray(b["seg_subdom"])
    e1s, e2s, e3s = np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]), np.asarray(b["seg_e3"])
    D_by, G_by = _material_by_section(json.loads(str(b["sections"])),
                                      json.loads(str(b["materials"])), center_ref=True)
    cents = nodes[quads].mean(1)
    k22_e = compute_k22(cents, e2s, e3s, quads)
    kg_e = compute_kg(cents, e1s, e2s, e3s, quads)        # hoop geodesic curvature (taper)

    rings = {}
    for side in ("L", "R"):
        rx = np.asarray(b["%s_x" % side]); rc = np.asarray(b["%s_cells" % side])
        rs = np.asarray(b["%s_subdom" % side]); re3 = np.asarray(b["%s_e3" % side])
        kr = compute_k22(rx[rc].mean(1), np.asarray(b["%s_e2" % side]), re3, rc)
        C6r, V0r, V1r = ring_general(rx, rc, rs, re3, D_by, G_by, kr, ax, list(cross), shear=shear)
        rings[side] = dict(C6=C6r, V0=V0r, V1=V1r)
        np.save(os.path.join(res_dir, "rm_%s_%s.npy" % (tg, side)), C6r)

    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment_general(
        nodes, quads, sd, e1s, e2s, e3s, D_by, G_by, k22_e, cross, shear=shear, kg_e=kg_e)
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
    Lz = float(nodes[:, ax].max() - nodes[:, ax].min())
    EB = (np.asarray(Dee) + V0.T @ Dhe) / Lz
    C, Psi = build_C_Psi_segment(nodes, quads, cross)
    Psi[3::5, 3] *= -1.0                       # general-op twist kernel: om1 = +1
    Dc = C.T
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl), jnp.array(Dll), jnp.array(Dle), jnp.array(Psi), jnp.array(Dc))
    bd1, bv1 = scatter("V1"); V1 = dirichlet_solve(Dhh, np.asarray(bb), bd1, bv1)
    S6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V1), jnp.array(V0), jnp.array(EB),
        jnp.array(np.asarray(V0DllV0) / Lz), jnp.array(np.asarray(DhlV0) / Lz),
        jnp.array(np.asarray(DhlTV0Dle) / Lz), jnp.array(Psi), jnp.array(Dc))
    S6 = 0.5 * (np.asarray(S6) + np.asarray(S6).T)
    np.save(os.path.join(res_dir, "rm_%s_seg.npy" % tg), S6)
    return rings["L"]["C6"], S6, rings["R"]["C6"]


def ana_iso(R, t, E=70e9, nu=0.3):
    G = E / (2 * (1 + nu)); A = 2 * math.pi * R * t; I = math.pi * R**3 * t
    return np.array([E * A, 0.5 * G * A, 0.5 * G * A, 2 * G * I, E * I, E * I])


# ------------------------------------------------------------------ commands
def cmd_gen():
    for regime in THICK:
        for mat in ("iso", "m45"):
            for aR in TAPERS:
                tg = gen_case(regime, mat, aR)
                print("mesh", tg)
    # orientation PNGs for the representative strongest-taper cases
    for regime in THICK:
        for mat in ("iso", "m45"):
            arrow_pngs(tag_of(regime, mat, 0.7))
    print("orientation PNGs ->", ORI)


def cmd_locking():
    """Thin prismatic tube: ring + taper 6x6 for every shear scheme vs analytic.
    Transverse-shear LOCKING shows as over-stiff C22/C33 (and C55/C66) for
    'full'; the MITC schemes must stay near the analytic values."""
    from segment_element_general import SHEAR_SCHEMES
    t = THICK["thin"]
    tg = tag_of("thin", "iso", 1.0)
    if not os.path.exists(os.path.join(MESH, "shell_%s.yaml" % tg)):
        gen_case("thin", "iso", 1.0)
    a = ana_iso(R0, t)
    print("THIN prismatic (t/R=%.2f) -- shear-scheme study vs analytic" % (t / R0))
    print("%-12s " % "scheme" + "  ".join("%7s" % k for k in LBL) + "   (ring | segment %err)")
    for sch in SHEAR_SCHEMES:
        rL, S6, rR = shell_solve(tg, shear=sch)
        dr = np.diag(rL); ds = np.diag(S6)
        print("%-12s " % sch + "  ".join("%+6.1f%%" % (100 * (dr[i] - a[i]) / a[i]) for i in range(6)) + "  | " +
              "  ".join("%+6.1f%%" % (100 * (ds[i] - a[i]) / a[i]) for i in range(6)))


def cmd_shell(regime, mat, shear="mitc4_both"):
    for aR in TAPERS:
        tg = tag_of(regime, mat, aR)
        rL, S6, rR = shell_solve(tg, shear=shear)
        print("shell %-18s C6diag(x1e9): L %s | seg %s | R %s"
              % (tg, np.array2string(np.diag(rL) / 1e9, precision=3),
                 np.array2string(np.diag(S6) / 1e9, precision=3),
                 np.array2string(np.diag(rR) / 1e9, precision=3)))


def cmd_tables(regime, mat):
    for aR in TAPERS:
        tg = tag_of(regime, mat, aR)
        fn = os.path.join(RES, "table_%s.dat" % tg)
        with open(fn, "w") as f:
            f.write("# %s : solid | RM(general) | %%err -- L boundary, taper, R boundary\n" % tg)
            for part, nm in (("L", "LEFT boundary"), ("seg", "TAPERED SEGMENT"), ("R", "RIGHT boundary")):
                Sh = np.load(os.path.join(RES, "rm_%s_%s.npy" % (tg, part)))
                So = np.load(os.path.join(RES, "solid_%s_%s.npy" % (tg, part)))
                Sh = 0.5 * (Sh + Sh.T); So = 0.5 * (So + So.T)
                f.write("\n== %s ==\n%-6s %15s %15s %9s\n" % (nm, "Cij", "solid", "RM", "%err"))
                thr = 1e-3 * max(abs(np.diag(So)).max(), abs(np.diag(Sh)).max())
                for i in range(6):
                    for j in range(i, 6):
                        if abs(So[i, j]) > thr or abs(Sh[i, j]) > thr:
                            e = 100 * (Sh[i, j] - So[i, j]) / So[i, j] if So[i, j] != 0 else float("nan")
                            f.write("C%d%d    %15.6e %15.6e %+8.1f%%\n" % (i + 1, j + 1, So[i, j], Sh[i, j], e))
        print("wrote", fn)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gen"
    if cmd == "gen":
        cmd_gen()
    elif cmd == "locking":
        cmd_locking()
    elif cmd == "shell":
        cmd_shell(sys.argv[2] if len(sys.argv) > 2 else "thin",
                  sys.argv[3] if len(sys.argv) > 3 else "iso",
                  sys.argv[4] if len(sys.argv) > 4 else "mitc4_both")
    elif cmd == "tables":
        cmd_tables(sys.argv[2] if len(sys.argv) > 2 else "thin",
                   sys.argv[3] if len(sys.argv) > 3 else "iso")
