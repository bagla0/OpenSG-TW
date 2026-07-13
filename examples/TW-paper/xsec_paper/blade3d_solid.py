"""blade3d_solid.py -- FULL-SOLID 3-D blade dehom field (not just the OML surface): loft the wall
through its thickness across the six windIO stations and store the field for ParaView slicing.

For each station the airfoil wall is parameterised (s = arc around the OML, t = 0..1 OML->IML using
the local laminate thickness from the shell YAML).  The VABS solid field (nearest Gauss/node in the
station .SM/.U) and the RM two-step recovery are evaluated on this (s,t) grid, then lofted across the
span into a structured volume:
  -> vtk/blade3d_solid.vtk   (StructuredGrid, arrays VABS_/OpenSG_RM_ {S11,S22,S12,u1,u2,u3}) -- slice
                              at any section in ParaView
  -> figures/blade3d_solid_S11.png   3-D render (outer surface + the six wall sections, thickness shown)
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IO = os.path.join(REPO, "third_party", "OpenSG_io")
sys.path.insert(0, REPO); sys.path.insert(0, IO); sys.path.insert(0, HERE)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm
from opensg_io import load_blade, build_cross_section

WINDIO = os.path.join(REPO, "examples", "data", "windio", "IEA-22-280-RWT.yaml")
D2 = os.path.join(REPO, "examples", "data", "2d_yaml"); VABS = os.path.join(D2, "IEA_VABS")
IB = os.path.join(REPO, "examples", "TW-paper", "iea22_blade", "data")
Y1 = os.path.join(REPO, "examples", "data", "1d_yaml", "IEA")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
VTK = os.path.join(HERE, "vtk"); os.makedirs(VTK, exist_ok=True)
CMAP = plt.cm.rainbow
N, NT, MPER = 140, 8, 10                              # arc pts, through-thickness layers, span interp
NW = 60                                              # points along each web mid-line
BLADE_LEN = 137.8                                    # IEA-22 blade length (m): span axis = r*BLADE_LEN
#                                                      (so the .vtk is not squashed span vs chord)

STATIONS = [("r020", 0.2000, os.path.join(IB, "shell_r020.yaml")),
            ("r0247", 0.2470, os.path.join(Y1, "shell_r0247.yaml")),
            ("r0399", 0.3993, os.path.join(Y1, "shell_r0399.yaml")),
            ("r0534", 0.5336, os.path.join(Y1, "shell_r0534.yaml")),
            ("r0739", 0.7389, os.path.join(Y1, "shell_r0739.yaml")),
            ("r0980", 0.9800, os.path.join(Y1, "shell_r0980.yaml"))]
COMP = ["S11", "S22", "S12", "u1", "u2", "u3"]


def row(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


def parse_glb(path):
    L = [ln.split() for ln in open(path).read().splitlines() if ln.strip()]
    a = [float(x) for x in L[4]]; b = [float(x) for x in L[5]]
    return np.array([a[0], b[0], b[1], a[1], a[2], a[3]])


def resample(xy, n):
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    c = np.vstack([xy, xy[0]])
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))]; d /= d[-1]
    t = np.linspace(0, 1, n, endpoint=False)
    return np.column_stack([np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])])


def wall_thickness(shell, contour):
    """Local laminate thickness at each contour point, from the shell-YAML layup (nearest skin elem)."""
    d = yaml.safe_load(open(shell))
    nd = np.array([row(n)[:2] for n in d["nodes"]])
    cells = np.array([[int(v) for v in row(e)] for e in d["elements"]]); cells -= cells.min()
    sect = {s["elementSet"]: sum(float(p[1]) for p in s["layup"]) for s in d["sections"]}
    et = np.zeros(len(cells))
    for grp in d["sets"]["element"]:
        if grp["name"] in sect:
            for lab in grp["labels"]:
                et[int(lab) - 1] = sect[grp["name"]]
    mid = nd[cells].mean(1)
    return et[cKDTree(mid).query(contour)[1]]


blade = load_blade(WINDIO)
rings = []                       # (r, P[N,NT,2], {comp: F[N,NT]})
print("station   r      wall t(mm) min/max   S11[min,max]")
for tag, r, shell in STATIONS:
    B = dehom_rm.build_rm_bundle(shell, ref="oml")
    FF = parse_glb(os.path.join(VABS, "iea_%s.sg.glb" % tag))
    cs = build_cross_section(blade, r=r)
    oml = resample(np.asarray(cs["nodes"], float), N)
    th = wall_thickness(shell, oml)
    cen = oml.mean(0)
    # inward unit normals along the contour
    tang = np.gradient(oml, axis=0); tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-30)
    nrm = np.column_stack([tang[:, 1], -tang[:, 0]])
    flip = ((cen - oml) * nrm).sum(1) < 0; nrm[flip] *= -1
    tt = np.linspace(0, 1, NT)
    P = oml[:, None, :] + tt[None, :, None] * (th[:, None, None] * nrm[:, None, :])   # (N,NT,2)
    pts2 = P.reshape(-1, 2)
    # VABS field at the (s,t) points
    U = np.loadtxt(os.path.join(VABS, "iea_%s.sg.U" % tag)); dsm = np.loadtxt(os.path.join(VABS, "iea_%s.sg.SM" % tag), skiprows=2)
    sm_xy, sm_s = dsm[:, :2], dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6
    ktg = cKDTree(sm_xy); ktn = cKDTree(U[:, 1:3])
    Vs = sm_s[ktg.query(pts2)[1]]; Vu = U[ktn.query(pts2)[1], 3:6] * 1e3
    # RM field at the (s,t) points
    Rs = np.asarray(dehom_rm.stress_at_points(B, pts2, beam_force_vabs=FF, frame="material", n_per_layer=4)["stress"]) / 1e6
    Ru = np.asarray(dehom_rm.disp_at_points(B, pts2, beam_force_vabs=FF)) * 1e3
    F = {}
    for j, c in enumerate(["S11", "S22", "S12"]):
        idx = [0, 1, 5][j]
        F["VABS_" + c] = Vs[:, idx].reshape(N, NT); F["RM_" + c] = Rs[:, idx].reshape(N, NT)
    for j, c in enumerate(["u1", "u2", "u3"]):
        F["VABS_" + c] = Vu[:, j].reshape(N, NT); F["RM_" + c] = Ru[:, j].reshape(N, NT)
    # --- webs: straight mid-line pa->pb through the interior (NW pts), VABS + RM fields ---
    # Identify each web by its chordwise distance from the LEADING EDGE (LE = min y2 on the OML),
    # normalised by chord.  Ordering webs by this gives a stable, geometric identity so the span
    # loft pairs the SAME physical web across stations -- robust to build order and to a station
    # having a different web count (the unmatched web is the one whose chord position has no
    # neighbour, i.e. a web that starts/stops = a web drop).
    tw = np.linspace(0, 1, NW); le_y2 = float(oml[:, 0].min())
    wlist = []
    for w in cs["webs"]:
        pa = np.asarray(cs["nodes"][w["a"]], float); pb = np.asarray(cs["nodes"][w["b"]], float)
        wp = pa[None, :] * (1 - tw[:, None]) + pb[None, :] * tw[:, None]              # (NW,2) web mid-line
        wVs = sm_s[ktg.query(wp)[1]]; wVu = U[ktn.query(wp)[1], 3:6] * 1e3
        wRs = np.asarray(dehom_rm.stress_at_points(B, wp, beam_force_vabs=FF, frame="material", n_per_layer=4)["stress"]) / 1e6
        wRu = np.asarray(dehom_rm.disp_at_points(B, wp, beam_force_vabs=FF)) * 1e3
        WF = {}
        for j, c in enumerate(["S11", "S22", "S12"]):
            idx = [0, 1, 5][j]
            WF["VABS_" + c] = wVs[:, idx]; WF["RM_" + c] = wRs[:, idx]
        for j, c in enumerate(["u1", "u2", "u3"]):
            WF["VABS_" + c] = wVu[:, j]; WF["RM_" + c] = wRu[:, j]
        xc = float((wp[:, 0].mean() - le_y2) / cs["chord"])       # chordwise position from LE (fraction)
        wlist.append(dict(name=w["name"], xc=xc, P=wp, F=WF))
    wlist.sort(key=lambda d: d["xc"])                            # order webs LE->TE (geometric identity)
    rings.append((r, P, F, wlist))
    print("  %-6s %.4f   %.1f / %.1f   webs=%d   [%6.2f,%6.2f]"
          % (tag, r, th.min() * 1e3, th.max() * 1e3, len(wlist), Vs[:, 0].min(), Vs[:, 0].max()))

# ---- loft (geometry + all fields, skin AND webs) across the span ----
span = []                                            # (r, P[N,NT,2], {field:[N,NT]}, [web dicts])
for k in range(len(rings) - 1):
    r0, P0, F0, W0 = rings[k]; r1, P1, F1, W1 = rings[k + 1]
    for m in range(MPER):
        t = m / MPER
        wln = [dict(name=wa["name"], P=(1 - t) * wa["P"] + t * wb["P"],       # web order fixed (web0,1,2)
                    F={kk: (1 - t) * wa["F"][kk] + t * wb["F"][kk] for kk in wa["F"]})
               for wa, wb in zip(W0, W1)]
        span.append((r0 + t * (r1 - r0), (1 - t) * P0 + t * P1,
                     {kk: (1 - t) * F0[kk] + t * F1[kk] for kk in F0}, wln))
span.append(rings[-1])
NS = len(span)

# ---- build the structured skin volume + web surfaces (physical span), merge, save one .vtk ----
try:
    import pyvista as pv
    X = np.zeros((N, NT, NS)); Y = np.zeros((N, NT, NS)); Z = np.zeros((N, NT, NS))
    for k, (rk, Pk, _F, _W) in enumerate(span):
        X[:, :, k] = rk * BLADE_LEN; Y[:, :, k] = Pk[:, :, 0]; Z[:, :, k] = Pk[:, :, 1]
    skin = pv.StructuredGrid(X, Y, Z)
    for key in span[0][2]:
        arr = np.zeros((N, NT, NS))
        for k in range(NS):
            arr[:, :, k] = span[k][2][key]
        skin.point_data[key] = arr.ravel(order="F")
    nw_ = len(span[0][3]); webgrids = []
    for wi in range(nw_):                            # each web = lofted mid-line surface (NW x NS)
        XW = np.zeros((NW, 1, NS)); YW = np.zeros((NW, 1, NS)); ZW = np.zeros((NW, 1, NS))
        for k in range(NS):
            wk = span[k][3][wi]
            XW[:, 0, k] = span[k][0] * BLADE_LEN; YW[:, 0, k] = wk["P"][:, 0]; ZW[:, 0, k] = wk["P"][:, 1]
        wg = pv.StructuredGrid(XW, YW, ZW)
        for key in span[0][3][wi]["F"]:
            arr = np.zeros((NW, 1, NS))
            for k in range(NS):
                arr[:, 0, k] = span[k][3][wi]["F"][key]
            wg.point_data[key] = arr.ravel(order="F")
        webgrids.append(wg)
    allb = skin.merge(webgrids) if webgrids else skin       # one unstructured .vtk: skin + 3 webs
    allb.save(os.path.join(VTK, "blade3d_solid.vtk"))
    print("wrote vtk/blade3d_solid.vtk  (skin %d pts + %d web surfaces, arrays: %s)"
          % (skin.n_points, nw_, ", ".join(span[0][2])))
except Exception as e:
    print("vtk save skipped:", repr(e)[:150])

# ---- 3-D render: outer surface lofted + the six wall sections filled (thickness visible) ----
def render(field, label, out):
    allf = np.concatenate([s[2][field].ravel() for s in span])
    m = np.nanpercentile(np.abs(allf), 99.5) or 1e-9; norm = Normalize(-m, m)
    fig = plt.figure(figsize=(15, 6)); ax = fig.add_subplot(111, projection="3d"); ax.set_proj_type("ortho")

    def pk(F2d):                                               # through-wall peak-magnitude value (signed)
        return F2d[np.arange(F2d.shape[0]), np.argmax(np.abs(F2d), axis=1)]
    Pks = [pk(s[2][field]) for s in span]                      # per span sample -> (N,)
    quads, cols = [], []
    for k in range(NS - 1):                                    # outer surface coloured by the PEAK through the wall
        rk, Pk = span[k][0], span[k][1]; rk1, Pk1 = span[k + 1][0], span[k + 1][1]
        for i in range(N):
            i1 = (i + 1) % N
            quads.append([(rk, Pk[i, 0, 0], Pk[i, 0, 1]), (rk, Pk[i1, 0, 0], Pk[i1, 0, 1]),
                          (rk1, Pk1[i1, 0, 0], Pk1[i1, 0, 1]), (rk1, Pk1[i, 0, 0], Pk1[i, 0, 1])])
            cols.append(0.25 * (Pks[k][i] + Pks[k][i1] + Pks[k + 1][i1] + Pks[k + 1][i]))
    for (rk, Pk, Fk, _Wk) in rings:                            # the six wall sections (through-thickness)
        for i in range(N):
            i1 = (i + 1) % N
            for j in range(NT - 1):
                quads.append([(rk, Pk[i, j, 0], Pk[i, j, 1]), (rk, Pk[i1, j, 0], Pk[i1, j, 1]),
                              (rk, Pk[i1, j + 1, 0], Pk[i1, j + 1, 1]), (rk, Pk[i, j + 1, 0], Pk[i, j + 1, 1])])
                cols.append(0.25 * (Fk[field][i, j] + Fk[field][i1, j] + Fk[field][i1, j + 1] + Fk[field][i, j + 1]))
    pc = Poly3DCollection(quads, facecolors=CMAP(norm(np.clip(cols, -m, m))), edgecolors="none", shade=False)
    ax.add_collection3d(pc)
    yall = np.concatenate([rr[1][:, :, 0].ravel() for rr in rings]); zall = np.concatenate([rr[1][:, :, 1].ravel() for rr in rings])
    ax.set_xlim(0.18, 1.0); ax.set_ylim(yall.min() - 0.3, yall.max() + 0.3); ax.set_zlim(zall.min() - 0.3, zall.max() + 0.3)
    ax.set_box_aspect((9.0, 5.0, 1.7)); ax.view_init(elev=20, azim=-72)
    ax.set_xlabel(r"spanwise position $r$", labelpad=12); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0)); ax.zaxis.line.set_color((1, 1, 1, 0))
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.0, aspect=18, location="right"); cb.set_label(label, fontsize=11)
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))


# full-solid blade: 3 displacement + 3 in-plane stress, each a SEPARATE figure (VABS solid field;
# the RM arrays are in blade3d_solid.vtk for the user's own ParaView views)
LAB = {"u1": r"$u_1$ through wall (mm)", "u2": r"$u_2$ through wall (mm)", "u3": r"$u_3$ through wall (mm)",
       "S11": r"$\sigma_{11}$ through wall (MPa)", "S22": r"$\sigma_{22}$ through wall (MPa)",
       "S12": r"$\sigma_{12}$ through wall (MPa)"}
for c in ["u1", "u2", "u3", "S11", "S22", "S12"]:
    render("VABS_" + c, "peak " + LAB[c], os.path.join(FIG, "blade3d_solid_%s.png" % c))
