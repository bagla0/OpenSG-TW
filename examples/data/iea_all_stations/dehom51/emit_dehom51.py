"""emit_dehom51.py -- run the RM two-step dehomogenization ONCE per station and EMIT VABS-like output
fields at the through-thickness gauss (sample) points, so post-processing can REUSE them instead of
re-running the dehom.  Uses the EXISTING mid-ref 1-D shell yamls (does NOT rebuild them).

Per station <tag>=iea_sNN, into out/dehom_out/ :
  <tag>.SM   y2 y3 c11 c12 c13 c22 c23 c33   RM local stress  [Pa]  (VABS material-frame column order)
  <tag>.EM   y2 y3 e11 e12 e13 e22 e23 e33   RM local strain
  <tag>.U    id y2 y3 u1 u2 u3               RM TOTAL local disp [m]  (warping + beam disp/rotation)
  <tag>.npz  P[N,NT,2], stress[N,NT,6], strain[N,NT,6], disp[N,NT,3], th[N], r   (fast reload cache)

Grid: N contour points x NT through-thickness layers (OML t=0 -> IML t=1), REFINED through the thickness.
"""
import os, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))                       # .../dehom51
ROOT = os.path.abspath(os.path.join(HERE, ".."))                       # .../iea_all_stations
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
IO = os.path.join(REPO, "third_party", "OpenSG_io")
XSEC = os.path.join(REPO, "examples", "TW-paper", "xsec_paper")
sys.path.insert(0, REPO); sys.path.insert(0, IO); sys.path.insert(0, XSEC)
import jax; jax.config.update("jax_enable_x64", True)
import yaml
from scipy.spatial import cKDTree
import dehom_rm
from opensg_io import load_blade, build_cross_section

WINDIO = os.path.join(ROOT, "IEA-22-280-RWT.yaml")
SHELLD = os.path.join(ROOT, "shell51", "1d_yaml")
FF_ALL = np.loadtxt(os.path.join(HERE, "beamdyn", "ff51_rmc_reform.dat"))
BD_OUT = os.path.join(HERE, "beamdyn", "iea51rmc_bd_driver.out")
OUT = os.path.join(HERE, "out", "dehom_out"); os.makedirs(OUT, exist_ok=True)
N, NT = 240, 16                                  # contour, REFINED through-thickness sample layers
VABS = [0, 5, 4, 1, 3, 2]                         # RM Voigt [11,22,33,23,13,12] -> VABS [11,12,13,22,23,33]


def resample(xy, n):
    if np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    c = np.vstack([xy, xy[0]])
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(c[:, 0]), np.diff(c[:, 1])))]; d /= d[-1]
    t = np.linspace(0, 1, n, endpoint=False)
    return np.column_stack([np.interp(t, d, c[:, 0]), np.interp(t, d, c[:, 1])])


def beam_kinematics(path, node):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    for i, l in enumerate(L):
        if l.strip().startswith("Time"):
            h = l.split(); r = np.array([rr.split() for rr in L[i + 2:]], float)[-1]
            g = lambda nm: r[h.index("N%03d_%s" % (node, nm))]
            TD = np.array([g("TDxr"), g("TDyr"), g("TDzr")]); RD = np.array([g("RDxr"), g("RDyr"), g("RDzr")])
            u_g = np.array([TD[2], -TD[1], TD[0]]); t1, t2, t3 = RD[2], -RD[1], RD[0]
            C = np.array([[1.0, -t3, t2], [t3, 1.0, -t1], [-t2, t1, 1.0]]); return u_g, C
    raise ValueError("no BeamDyn header")


def _row(v):
    return [float(x) for x in (v[0].split() if isinstance(v, list) and isinstance(v[0], str) else v)]


def wall_thickness(shell, contour):
    d = yaml.safe_load(open(shell))
    nd = np.array([_row(n)[:2] for n in d["nodes"]])
    cells = np.array([[int(x) for x in _row(e)] for e in d["elements"]]); cells -= cells.min()
    sect = {s["elementSet"]: sum(float(p[1]) for p in s["layup"]) for s in d["sections"]}
    et = np.zeros(len(cells))
    for grp in d["sets"]["element"]:
        if grp["name"] in sect:
            for lab in grp["labels"]:
                et[int(lab) - 1] = sect[grp["name"]]
    m = nd[cells].mean(1)
    return et[cKDTree(m).query(contour)[1]]


blade = load_blade(WINDIO)
print("emitting RM dehom output (%d contour x %d thickness) -> %s" % (N, NT, os.path.relpath(OUT, HERE)))
for i in range(51):
    shell = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    if not os.path.exists(shell):
        print("  s%02d missing" % i); continue
    tag = "iea_s%02d" % i; r = i / 50.0
    try:
        B = dehom_rm.build_rm_bundle(shell); FF = FF_ALL[i, 1:]
        oml = resample(np.asarray(build_cross_section(blade, r=r)["nodes"], float), N)
        th = wall_thickness(shell, oml)
        tg = np.gradient(oml, axis=0); tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-30)
        nrm = np.column_stack([tg[:, 1], -tg[:, 0]])
        cen = oml.mean(0); flip = ((cen - oml) * nrm).sum(1) < 0; nrm[flip] *= -1
        tt = np.linspace(0, 1, NT)
        P = oml[:, None, :] + tt[None, :, None] * (th[:, None, None] * nrm[:, None, :])   # (N,NT,2) OML->IML
        pts = P.reshape(-1, 2)
        res = dehom_rm.stress_at_points(B, pts, beam_force_vabs=FF, frame="material", n_per_layer=4)
        S = np.asarray(res["stress"]); E = np.asarray(res["strain"])                        # (M,6) Voigt
        W = np.asarray(dehom_rm.disp_at_points(B, pts, beam_force_vabs=FF))
        u_g, C = beam_kinematics(BD_OUT, i + 1)
        r3 = np.column_stack([np.zeros(len(pts)), pts[:, 0], pts[:, 1]])
        U = u_g + (C @ (W + r3).T).T - r3                                                   # total local disp (m)
        # ---- VABS-like text files ----
        np.savetxt(os.path.join(OUT, tag + ".SM"), np.column_stack([pts, S[:, VABS]]), fmt="%.6e",
                   header="y2 y3 c11 c12 c13 c22 c23 c33  RM local stress [Pa] (VABS order); %dx%d grid" % (N, NT))
        np.savetxt(os.path.join(OUT, tag + ".EM"), np.column_stack([pts, E[:, VABS]]), fmt="%.6e",
                   header="y2 y3 e11 e12 e13 e22 e23 e33  RM local strain")
        ids = np.arange(1, len(pts) + 1)
        np.savetxt(os.path.join(OUT, tag + ".U"), np.column_stack([ids, pts, U]),
                   fmt=["%d", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e"],
                   header="id y2 y3 u1 u2 u3  RM TOTAL local disp [m]")
        # ---- fast reload cache ----
        np.savez(os.path.join(OUT, tag + ".npz"), P=P, stress=S.reshape(N, NT, 6),
                 strain=E.reshape(N, NT, 6), disp=U.reshape(N, NT, 3), th=th, r=r)
        print("  s%02d r=%.3f  emitted .SM/.EM/.U/.npz  (%d pts, |S11|max %.1f MPa)"
              % (i, r, len(pts), np.abs(S[:, 0]).max() / 1e6))
    except Exception as e:
        print("  s%02d FAIL: %s" % (i, str(e)[:70]))
print("done -> out/dehom_out/  (reuse these for contour plots + full-blade post-processing)")
