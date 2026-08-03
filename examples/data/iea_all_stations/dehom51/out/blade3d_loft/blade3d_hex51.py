"""blade3d_hex51.py -- FULL IEA-22 blade as a lofted HEX solid of the wall, coloured by the RM-shell
dehomogenized fields, for ParaView.

Pipeline (one station at a time, then loft along the span):
  * geometry : for each of the 51 span stations we take the OML airfoil contour (windIO) resampled to
               N=160 points and build a structured through-thickness wall grid -- NT=8 node layers from
               the OML (t=0) to the IML (t=1) along the inward normal, using the LOCAL laminate thickness
               from the station's center-ref shell YAML.  Between adjacent stations we interpolate MPER=14
               span planes, so the wall is a clean (N x NT x NS) structured hex volume.
  * fields   : at every (contour, depth) point we recover the RM two-step dehom stress (material frame)
               and the TOTAL local displacement u = u_g + C(w+r) - r (warping + beam disp/rotation from
               the RM BeamDyn node), under each station's own beam load FF.  VABS .SM/.U are sampled at
               the same points for a side-by-side array.
  * output   : vtk/blade3d_hex51.vtk  (pyvista StructuredGrid) with point arrays
                   RM_{S11,S22,S12,u1,u2,u3}, VABS_{...}, and the 3-vector RM_disp (m) for WarpByVector.
               Render 7 PNGs with ParaView: 6 field colourings + 1 deformed shape (see render_blade_pv.py).

  N=160 contour, NT=8 through-thickness, MPER=14 span/interval  (per the request).
"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
import yaml
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ---- locate the repo + station data (this script lives in dehom51/out/blade3d_loft/) --------------
HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))                 # .../iea_all_stations/dehom51
ROOT = os.path.abspath(os.path.join(IEA, ".."))                       # .../iea_all_stations
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))          # OpenSG-TW-claude
IO = os.path.join(REPO, "third_party", "OpenSG_io")
XSEC = os.path.join(REPO, "examples", "TW-paper", "xsec_paper")
sys.path.insert(0, REPO); sys.path.insert(0, IO); sys.path.insert(0, XSEC)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm
from opensg_io import load_blade, build_cross_section

WINDIO = os.path.join(ROOT, "IEA-22-280-RWT.yaml")
SHELLD = os.path.join(ROOT, "shell51", "1d_yaml")
VABS = os.path.join(IEA, "out", "VABS_iea51")
FF_ALL = np.loadtxt(os.path.join(IEA, "beamdyn", "ff51_rmc_reform.dat"))   # 51 rows: eta F1..M3
BD_OUT = os.path.join(IEA, "beamdyn", "iea51rmc_bd_driver.out")
VTK = os.path.join(HERE, "vtk"); os.makedirs(VTK, exist_ok=True)

N, NT, MPER = 240, 8, 20                       # contour pts, through-thickness node layers, span/interval
#                                                (refined from 160/14 for smoother contour + span interpolation)
BLADE_LEN = 138.204                            # m, so the span axis is not squashed vs chord
COMP = ["S11", "S22", "S12", "u1", "u2", "u3"]
SVOIGT = {"S11": 0, "S22": 1, "S12": 5}        # RM/VABS Voigt index for the 3 in-plane stresses


def _row(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


def resample(xy, n):
    """Resample a closed contour to n points by normalized arc length."""
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    c = np.vstack([xy, xy[0]])
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))]; d /= d[-1]
    t = np.linspace(0, 1, n, endpoint=False)
    return np.column_stack([np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])])


def wall_thickness(shell, contour):
    """Local laminate thickness (m) at each contour point, from the shell-YAML layup (nearest skin elem)."""
    d = yaml.safe_load(open(shell))
    nd = np.array([_row(n)[:2] for n in d["nodes"]])
    cells = np.array([[int(x) for x in _row(e)] for e in d["elements"]]); cells -= cells.min()
    sect = {s["elementSet"]: sum(float(p[1]) for p in s["layup"]) for s in d["sections"]}
    et = np.zeros(len(cells))
    for grp in d["sets"]["element"]:
        if grp["name"] in sect:
            for lab in grp["labels"]:
                et[int(lab) - 1] = sect[grp["name"]]
    mid = nd[cells].mean(1)
    return et[cKDTree(mid).query(contour)[1]]


def beam_kinematics(path, node):
    """beam translation u_g + linearized rotation C (VABS frame) at a BeamDyn output node (== VABS .U convention)."""
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith("Time"):
            h = l.split(); r = np.array([rr.split() for rr in L[i + 2:]], float)[-1]
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")]); RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]]); t1, t2, t3 = RD[2], -RD[1], RD[0]
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]]); return u_g, C
    raise ValueError("no BeamDyn header")


blade = load_blade(WINDIO)
rings = []                       # (r, P[N,NT,2], {comp: F[N,NT]})
fails = []
print("station   r      wall t(mm) min/max   RM S11[min,max] MPa   |u|max m")
for i in range(51):
    shell = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    smp = os.path.join(VABS, "iea_s%02d.sg.SM" % i); up = os.path.join(VABS, "iea_s%02d.sg.U" % i)
    if not (os.path.exists(shell) and os.path.exists(smp) and os.path.exists(up)):
        fails.append((i, "missing")); continue
    try:
        r = i / 50.0
        B = dehom_rm.build_rm_bundle(shell)                    # ref auto-read from the yaml (center)
        FF = FF_ALL[i, 1:]
        oml = resample(np.asarray(build_cross_section(blade, r=r)["nodes"], float), N)   # OML contour (N,2)
        th = wall_thickness(shell, oml)                        # local wall thickness (N,)
        cen = oml.mean(0)
        tang = np.gradient(oml, axis=0); tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-30)
        nrm = np.column_stack([tang[:, 1], -tang[:, 0]])       # unit normal
        flip = ((cen - oml) * nrm).sum(1) < 0; nrm[flip] *= -1  # make it point INWARD (OML -> IML)
        tt = np.linspace(0, 1, NT)
        P = oml[:, None, :] + tt[None, :, None] * (th[:, None, None] * nrm[:, None, :])   # (N,NT,2) wall nodes
        pts2 = P.reshape(-1, 2)
        # RM two-step dehom (material frame) + TOTAL local disp at every wall point
        S = np.asarray(dehom_rm.stress_at_points(B, pts2, beam_force_vabs=FF, frame="material", n_per_layer=4)["stress"]) / 1e6
        W = np.asarray(dehom_rm.disp_at_points(B, pts2, beam_force_vabs=FF))
        u_g, C = beam_kinematics(BD_OUT, i + 1)
        r3 = np.column_stack([np.zeros(len(pts2)), pts2[:, 0], pts2[:, 1]])
        U = u_g + (C @ (W + r3).T).T - r3                      # total recovered local disp (m)
        # VABS .SM/.U at the same points
        dsm = np.loadtxt(smp, skiprows=2); Uv = np.loadtxt(up)
        sm_xy = dsm[:, :2]; sm_s = dsm[:, 2:8][:, [0, 3, 5, 4, 2, 1]] / 1e6   # -> Voigt [11,22,33,23,13,12]
        Vs = sm_s[cKDTree(sm_xy).query(pts2)[1]]
        Vu = Uv[cKDTree(Uv[:, 1:3]).query(pts2)[1], 3:6]
        F = {}
        for c in ("S11", "S22", "S12"):
            F["RM_" + c] = S[:, SVOIGT[c]].reshape(N, NT); F["VABS_" + c] = Vs[:, SVOIGT[c]].reshape(N, NT)
        for j, c in enumerate(("u1", "u2", "u3")):
            F["RM_" + c] = U[:, j].reshape(N, NT); F["VABS_" + c] = Vu[:, j].reshape(N, NT)
        rings.append((r, P, F))
        print("  s%02d  %.3f   %5.1f / %5.1f   [%7.2f,%7.2f]   %.3f"
              % (i, r, th.min() * 1e3, th.max() * 1e3, S[:, 0].min(), S[:, 0].max(), np.linalg.norm(U, axis=1).max()))
    except Exception as e:
        fails.append((i, str(e)[:70])); print("  s%02d FAIL: %s" % (i, str(e)[:70]))

print("\n%d/51 stations lofted ; fails: %s" % (len(rings), fails))

# ---- loft geometry + all fields across the span (MPER interpolated planes per station interval) ----
span = []                        # (r, P[N,NT,2], {field:[N,NT]})
for k in range(len(rings) - 1):
    r0, P0, F0 = rings[k]; r1, P1, F1 = rings[k + 1]
    for m in range(MPER):
        t = m / MPER
        span.append((r0 + t * (r1 - r0), (1 - t) * P0 + t * P1,
                     {kk: (1 - t) * F0[kk] + t * F1[kk] for kk in F0}))
span.append(rings[-1])
NS = len(span)

# ---- structured hex volume (physical span) -> one .vtk with all arrays + a disp vector for warping ----
import pyvista as pv
X = np.zeros((N, NT, NS)); Y = np.zeros((N, NT, NS)); Z = np.zeros((N, NT, NS))
for k, (rk, Pk, _F) in enumerate(span):
    X[:, :, k] = rk * BLADE_LEN; Y[:, :, k] = Pk[:, :, 0]; Z[:, :, k] = Pk[:, :, 1]
grid = pv.StructuredGrid(X, Y, Z)
for key in span[0][2]:
    arr = np.zeros((N, NT, NS))
    for k in range(NS):
        arr[:, :, k] = span[k][2][key]
    grid.point_data[key] = arr.ravel(order="F")
# 3-vector RM displacement (m) in the mesh frame (X=span<-u1, Y=y2<-u2, Z=y3<-u3) for WarpByVector
grid.point_data["RM_disp"] = np.column_stack([grid.point_data["RM_u1"], grid.point_data["RM_u2"],
                                              grid.point_data["RM_u3"]])
out = os.path.join(VTK, "blade3d_hex51.vtk")
grid.save(out)
print("wrote %s  (%d pts, %d hex, arrays: %s + RM_disp[vec])"
      % (os.path.relpath(out, HERE), grid.n_points, (N) * (NT - 1) * (NS - 1), ", ".join(sorted(span[0][2]))))
