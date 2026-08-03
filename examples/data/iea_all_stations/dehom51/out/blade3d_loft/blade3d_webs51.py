"""blade3d_webs51.py -- lofted SHEAR-WEB ribbons of the IEA-22 blade, coloured by the RM dehomogenized
fields, saved as vtk/blade3d_webs51.vtk (pyvista PolyData quads).  COMPANION to blade3d_hex51.py, which
lofts only the OML skin wall -- this adds the internal shear webs so the 3-D render shows them.

Each web (from blade.webs_at) is the straight mid-line from its OML start-arc to its end-arc; it is lofted
along the span over the contiguous run of stations where that web exists, with MPER interpolated planes per
interval.  RM two-step dehom stress (material frame) + TOTAL local disp are evaluated on the web mid-line at
each station (same bundle/loads as the skin), so the webs colour consistently with blade3d_hex51.vtk.
"""
import os, sys
import numpy as np
from collections import defaultdict
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
IEA = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(IEA, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
IO = os.path.join(REPO, "third_party", "OpenSG_io")
XSEC = os.path.join(REPO, "examples", "TW-paper", "xsec_paper")
sys.path.insert(0, REPO); sys.path.insert(0, IO); sys.path.insert(0, XSEC)
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm
from opensg_io import load_blade, build_cross_section
import pyvista as pv

WINDIO = os.path.join(ROOT, "IEA-22-280-RWT.yaml")
SHELLD = os.path.join(ROOT, "shell51", "1d_yaml")
FF_ALL = np.loadtxt(os.path.join(IEA, "beamdyn", "ff51_rmc_reform.dat"))
BD_OUT = os.path.join(IEA, "beamdyn", "iea51rmc_bd_driver.out")
VTK = os.path.join(HERE, "vtk"); os.makedirs(VTK, exist_ok=True)
N, NW, MPER = 240, 12, 20
BLADE_LEN = 138.204
SVOIGT = {"S11": 0, "S22": 1, "S12": 5}
FIELDS = ["RM_S11", "RM_S22", "RM_S12", "RM_u1", "RM_u2", "RM_u3"]


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


blade = load_blade(WINDIO)
web_st = defaultdict(dict)                       # name -> {i: (r, Pw[NW,2], {field:[NW]})}
for i in range(51):
    shell = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    if not os.path.exists(shell):
        continue
    r = i / 50.0
    try:
        webs = blade.webs_at(r)
        if not webs:
            print("  s%02d r=%.3f  no webs" % (i, r)); continue
        B = dehom_rm.build_rm_bundle(shell); FF = FF_ALL[i, 1:]
        oml = resample(np.asarray(build_cross_section(blade, r=r)["nodes"], float), N)
        u_g, C = beam_kinematics(BD_OUT, i + 1)
        for w in webs:
            ia = int(round(w["s"] * N)) % N; ib = int(round(w["e"] * N)) % N
            Pa = oml[ia]; Pb = oml[ib]
            Pw = Pa[None, :] + np.linspace(0, 1, NW)[:, None] * (Pb - Pa)[None, :]
            Sw = np.asarray(dehom_rm.stress_at_points(B, Pw, beam_force_vabs=FF, frame="material", n_per_layer=4)["stress"]) / 1e6
            Ww = np.asarray(dehom_rm.disp_at_points(B, Pw, beam_force_vabs=FF))
            r3 = np.column_stack([np.zeros(NW), Pw[:, 0], Pw[:, 1]])
            Uw = u_g + (C @ (Ww + r3).T).T - r3
            F = {}
            for c in ("S11", "S22", "S12"):
                F["RM_" + c] = Sw[:, SVOIGT[c]]
            for j, c in enumerate(("u1", "u2", "u3")):
                F["RM_" + c] = Uw[:, j]
            web_st[w["name"]][i] = (r, Pw, F)
        print("  s%02d r=%.3f  webs=%s" % (i, r, [w["name"] for w in webs]))
    except Exception as e:
        print("  s%02d FAIL: %s" % (i, str(e)[:70]))

# ---- loft each web over its contiguous station runs -> quad PolyData ----
allpts = []; allfaces = []; fld = {k: [] for k in FIELDS}; disp = []
base = 0
for name, d in web_st.items():
    idxs = sorted(d)
    runs, run = [], [idxs[0]]
    for a, b in zip(idxs, idxs[1:]):
        if b == a + 1:
            run.append(b)
        else:
            runs.append(run); run = [b]
    runs.append(run)
    for run in runs:
        if len(run) < 2:
            continue
        planes = []
        for k in range(len(run) - 1):
            r0, P0, F0 = d[run[k]]; r1, P1, F1 = d[run[k + 1]]
            for m in range(MPER):
                t = m / MPER
                planes.append(((r0 + t * (r1 - r0)) * BLADE_LEN, (1 - t) * P0 + t * P1,
                               {kk: (1 - t) * F0[kk] + t * F1[kk] for kk in F0}))
        rL, PL, FL = d[run[-1]]; planes.append((rL * BLADE_LEN, PL, FL))
        NP = len(planes)
        for X, Pw, F in planes:
            for n in range(NW):
                allpts.append([X, Pw[n, 0], Pw[n, 1]])
                for kk in FIELDS:
                    fld[kk].append(float(F[kk][n]))
                disp.append([float(F["RM_u1"][n]), float(F["RM_u2"][n]), float(F["RM_u3"][n])])
        for p in range(NP - 1):
            for n in range(NW - 1):
                a = base + p * NW + n
                allfaces += [4, a, a + 1, a + NW + 1, a + NW]
        base += NP * NW

mesh = pv.PolyData(np.array(allpts), np.array(allfaces))
for kk in FIELDS:
    mesh.point_data[kk] = np.array(fld[kk])
mesh.point_data["RM_disp"] = np.array(disp)
out = os.path.join(VTK, "blade3d_webs51.vtk")
mesh.save(out)
print("\nwrote %s  (%d pts, %d web quads, webs=%s)"
      % (os.path.relpath(out, HERE), mesh.n_points, mesh.n_cells, list(web_st)))
