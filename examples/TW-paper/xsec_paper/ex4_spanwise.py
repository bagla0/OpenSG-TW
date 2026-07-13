"""ex4_spanwise.py -- EXAMPLE 4 on the REAL windIO stations + spanwise local-field recovery.

Airfoil-definition stations (windIO spanwise_position, 4dp = the VABS .sg labels), plus the
r=0.2 root example station:
    r020  0.2000  (root example, between FFA-W3-360 and -330blend)
    r0247 0.2470  FFA-W3-330blend
    r0399 0.3993  FFA-W3-301
    r0534 0.5336  FFA-W3-270blend
    r0739 0.7389  FFA-W3-241
    r0980 0.9800  FFA-W3-211

For each station:
  * build the 1-D RM shell live from windIO (OpenSG_io build_cross_section, OML contour
    fraction=0.0), homogenize with the RM 6-DOF ring (ring_indep, mitc4_g23, OML) and compare
    the Timoshenko 6x6 against the VABS .K (2-D solid, the SAME meshes as the dehom);
  * dehomogenize under the station's own VABS beam load (parsed from the .glb) at the section
    top-most point (max y3) and compare the RM local stress/displacement against VABS .SM/.U.

  -> examples/data/1d_yaml/IEA/shell_r0XXX.yaml         (built shells at odd stations)
     results/ex4_spanwise.npz                            (stiffness %err + spanwise fields)
"""
import os, sys, time
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
MITC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "mitc_rm_segment"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IO = os.path.join(REPO, "third_party", "OpenSG_io")
for q in (MITC, REPO, IO, HERE):
    if q not in sys.path:
        sys.path.insert(0, q)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax; jax.config.update("jax_enable_x64", True)
import yaml
from segment_element import compute_k22
from solve_segment_jax import _material_by_section
from run_ring_indep import ring_indep
from xsec_5v6_master import _row, _norm_materials, LBL
from opensg_io import load_blade, build_cross_section, emit_opensg_yaml
import dehom_rm

WINDIO = os.path.join(REPO, "examples", "data", "windio", "IEA-22-280-RWT.yaml")
D2 = os.path.join(REPO, "examples", "data", "2d_yaml")
VABS = os.path.join(D2, "IEA_VABS")
Y1 = os.path.join(REPO, "examples", "data", "1d_yaml", "IEA")
IB = os.path.abspath(os.path.join(HERE, "..", "iea22_blade", "data"))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
os.makedirs(Y1, exist_ok=True)

STATIONS = [("r020", 0.2000, "FFA-W3-360/330"), ("r0247", 0.2470, "FFA-W3-330blend"),
            ("r0399", 0.3993, "FFA-W3-301"), ("r0534", 0.5336, "FFA-W3-270blend"),
            ("r0739", 0.7389, "FFA-W3-241"), ("r0980", 0.9800, "FFA-W3-211")]


def load_ring_ref(path, ref="oml"):
    d = yaml.safe_load(open(path))
    rx = np.array([_row(r)[:3] for r in d["nodes"]], dtype=float)
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], dtype=int)
    if cells.min() == 1:
        cells = cells - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], dtype=float)
    re3 = ori[:, 6:9]
    sections = d["sections"]; materials = _norm_materials(d["materials"])
    setname_to_sec = {s["elementSet"]: i for i, s in enumerate(sections)}
    rsub = np.zeros(len(cells), dtype=int)
    for grp in d["sets"]["element"]:
        si = setname_to_sec[grp["name"]]
        for lab in grp["labels"]:
            rsub[int(lab) - 1] = si
    D_by, G_by = _material_by_section(sections, materials, center_ref=(ref == "center"))
    k22 = compute_k22(rx[cells].mean(1), ori[:, 3:6], re3, cells)
    return dict(rx=rx, cells=cells, rsub=rsub, re3=re3, D_by=D_by, G_by=G_by,
                k22=k22, ax=2, cross=[0, 1])


def c6(R):
    C = ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], R["G_by"],
                   R["k22"], R["ax"], R["cross"], shear="mitc4_g23", lam_space="elem")
    return 0.5 * (C + C.T)


def load_vabs_timo(path):
    L = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(L) if "Timoshenko Stiffness Matrix" in ln)
    rows = []
    for ln in L[i + 1:]:
        p = ln.split()
        try:
            [float(x) for x in p]; ok = (len(p) == 6)
        except ValueError:
            ok = False
        if ok:
            rows.append([float(x) for x in p])
        if len(rows) == 6:
            break
    return np.array(rows)


def parse_glb(path):
    """VABS .glb beam load -> FF in VABS order [F1,F2,F3,M1,M2,M3].
    Line 5 = F1 M1 M2 M3 ; line 6 = F2 F3."""
    L = [ln.split() for ln in open(path).read().splitlines() if ln.strip()]
    a = [float(x) for x in L[4]]; b = [float(x) for x in L[5]]
    return np.array([a[0], b[0], b[1], a[1], a[2], a[3]])


def load_sm(path):
    d = np.loadtxt(path, skiprows=2)
    return d[:, :2], d[:, 2:8][:, [0, 3, 5, 4, 2, 1]]     # xy, [S11,S22,S33,S23,S13,S12]


# =================================================================================
blade = load_blade(WINDIO)
recs = []
print("=== build shell (OML) + RM 6x6 vs VABS .K + dehom-at-top per station ===", flush=True)
for tag, r, af in STATIONS:
    if tag == "r020":
        shell = os.path.join(IB, "shell_r020.yaml")                  # bundled OML shell (validated)
    else:
        shell = os.path.join(Y1, "shell_%s.yaml" % tag)
        if not os.path.exists(shell):
            cs = build_cross_section(blade, r=r, mesh_size=0.01)
            emit_opensg_yaml(cs, shell, fraction=0.0)                # OML contour, like shell_r020
    vt = "iea_%s" % tag
    Kf = os.path.join(VABS, "%s.sg.K" % vt)
    Uf = os.path.join(VABS, "%s.sg.U" % vt)
    SMf = os.path.join(VABS, "%s.sg.SM" % vt)

    t0 = time.time()
    B = dehom_rm.build_rm_bundle(shell, ref="oml")                   # RM ring: 6x6 + warping (one solve)
    C = 0.5 * (B["Timo"] + B["Timo"].T)
    t_h = time.time() - t0
    S = 0.5 * (load_vabs_timo(Kf) + load_vabs_timo(Kf).T)
    diag = np.array([100.0 * (C[i, i] - S[i, i]) / S[i, i] for i in range(6)])
    frob = np.linalg.norm(C - S) / np.linalg.norm(S) * 100.0

    # ---- dehom at the section top-most point (max y3), under this station's VABS load ----
    U = np.loadtxt(Uf); sm_xy, sm_s = load_sm(SMf)
    FF = parse_glb(os.path.join(VABS, "%s.sg.glb" % vt))
    top = int(np.argmax(U[:, 2])); pt = U[top, 1:3]
    uv = U[top, 3:6]                                                  # VABS disp (m)
    sv = sm_s[cKDTree(sm_xy).query(pt)[1]]                            # VABS stress (Pa) [S11,S22,S33,S23,S13,S12]
    ur = np.asarray(dehom_rm.disp_at_points(B, [pt], beam_force_vabs=FF))[0]
    sr = np.asarray(dehom_rm.stress_at_points(B, [pt], beam_force_vabs=FF, frame="material")["stress"])[0]
    recs.append(dict(tag=tag, r=r, af=af, diag=diag, frob=frob, t_h=t_h,
                     pt=pt, uv=uv, ur=ur, sv=sv, sr=sr, FF=FF))
    print("r=%.4f %-6s Frob=%5.1f%%  diag[%s]  | top y=(%.3f,%.3f) "
          "s11 V/RM=%.2f/%.2f MPa  |u| V/RM=%.3f/%.3f mm  [%.1fs]"
          % (r, tag, frob, " ".join("%s%+5.1f" % (LBL[i], diag[i]) for i in range(6)),
             pt[0], pt[1], sv[0] / 1e6, sr[0] / 1e6,
             np.linalg.norm(uv) * 1e3, np.linalg.norm(ur) * 1e3, t_h), flush=True)

# ---- save ----
rr = np.array([x["r"] for x in recs])
np.savez(os.path.join(OUT, "ex4_spanwise.npz"),
         r=rr, tags=[x["tag"] for x in recs], af=[x["af"] for x in recs], labels=LBL,
         diag_err=np.array([x["diag"] for x in recs]), frob=np.array([x["frob"] for x in recs]),
         pt=np.array([x["pt"] for x in recs]),
         uv=np.array([x["uv"] for x in recs]), ur=np.array([x["ur"] for x in recs]),
         sv=np.array([x["sv"] for x in recs]), sr=np.array([x["sr"] for x in recs]),
         FF=np.array([x["FF"] for x in recs]))
print("\nwrote results/ex4_spanwise.npz  (%d stations)" % len(recs))
