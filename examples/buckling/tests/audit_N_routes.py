"""audit_N_routes.py -- AUDIT (read-only) of the pre-buckling membrane resultant N recovery.

Route 1 "direct"   : N = A @ s6[0:3] + B @ s6[3:6]   (emitted mid-ref ABD + RM dehom shell strains)
Route 2 "integral" : N = int sigma_11 dz             (through-thickness integral of the 2-step 3-D stress)

Task 1: sweep stations, every distinct layup, ratio(integral/direct); label monolithic vs sandwich.
Task 2: bug 3 -- int A11 ds  vs  EA = C6[0,0]  (and the hoop-free reduced form  int (A11-A12^2/A22) ds).
Task 3: cost of route 2 for one full station (all section elements), batched.

Nothing here writes to / changes production code.
"""
import os, sys, time, pickle
import numpy as np

ROOT = "/home/roger/a/bagla0/OpenSG-TW-claude"
XSEC = os.path.join(ROOT, "examples", "TW-paper", "xsec_paper")
IEA = os.path.join(ROOT, "examples", "data", "iea_all_stations")
BUCK = os.path.join(ROOT, "examples", "buckling")
CACHE = os.path.join(BUCK, "data", "blade_cache")
sys.path.insert(0, ROOT); sys.path.insert(0, XSEC); sys.path.insert(0, BUCK)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax; jax.config.update("jax_enable_x64", True)
import yaml
import dehom_rm
from emit_abd import load_station_abd
from opensg_jax.fe_jax.msg_materials import compute_ABD_matrix, plate_stress_at_depth

FF = [1.0e6, 0.0, 0.0, 0.0, 5.0e7, 0.0]        # axial + flap moment, VABS order
STATIONS = [5, 10, 20, 30, 40]
CORE_E = 2.0e9                                  # E1 below this = "low-modulus core" material
CORE_FRAC = 0.25                                # ply thicker than this fraction of the wall = "thick"


def shell_path(i):
    return os.path.join(IEA, "shell51", "1d_yaml", "iea_s%02d_shell.yaml" % i)


def abd_path(i):
    return os.path.join(IEA, "dehom51", "out", "abd", "iea_s%02d_abd.yaml" % i)


def bundle(i):
    cf = os.path.join(CACHE, "bundle_s%02d.pkl" % i)
    if os.path.exists(cf):
        return pickle.load(open(cf, "rb"))
    B = dehom_rm.build_rm_bundle(shell_path(i))
    pickle.dump(B, open(cf, "wb"))
    return B


def mat_E1(i):
    d = yaml.safe_load(open(shell_path(i)))
    return {m["name"]: float(m["elastic"]["E"][0]) for m in d["materials"]}


def classify(layup_raw, E1):
    """(kind, core_frac, h, nply) from the emitted abd-yaml 'plies' list."""
    h = float(layup_raw["thickness"]); pl = layup_raw["plies"]
    cf = 0.0
    for p in pl:
        if E1.get(p["material"], 1e12) < CORE_E:
            cf += float(p["thickness"])
    cf = cf / h if h > 0 else 0.0
    kind = "SANDWICH" if cf > CORE_FRAC else ("core-thin" if cf > 1e-9 else "monolithic")
    return kind, cf, h, len(pl)


# ---------- route 2 machinery (identical math to dehom_rm.stress_at_points, plate level) ----------
def warp_for(B, name, n_per_layer=4, elem_order=2):
    inf = B["layup_db"][name]
    return compute_ABD_matrix(inf["thick"], inf["angles"], inf["mat_names"], B["material_db"],
                              n_per_layer=n_per_layer, return_warping=True,
                              elem_order=elem_order)[2]


def N_integral_plywise(warp, s6, frac, plies, ng=16):
    """int sigma dz with EXACT per-ply Gauss quadrature (no quadrature error at ply jumps).
    s6 is at the ring reference (frac*h from the OML); shift to OML exactly as stress_at_points does."""
    h = float(warp["node_x"][-1])
    s6r = np.array(s6, float); s6r[0:3] = s6r[0:3] - frac * h * s6r[3:6]
    xg, wg = np.polynomial.legendre.leggauss(ng)
    acc = np.zeros(6); z0 = 0.0
    for p in plies:
        t = float(p["thickness"]); z1 = z0 + t
        zs = 0.5 * (z1 - z0) * xg + 0.5 * (z1 + z0)
        for z, w in zip(zs, wg):
            _, Sig, _ = plate_stress_at_depth(warp, s6r, float(z))
            acc += w * 0.5 * t * np.asarray(Sig, float)
        z0 = z1
    return acc                                    # laminate Voigt [s11,s22,s33,g23,g13,s12]


def N_integral_uniform(warp, s6, frac, npts=41):
    """int sigma dz with the ORIGINAL probe's uniform trapz sampling across the wall."""
    h = float(warp["node_x"][-1])
    s6r = np.array(s6, float); s6r[0:3] = s6r[0:3] - frac * h * s6r[3:6]
    zs = np.linspace(0.0, h, npts)
    S = np.array([np.asarray(plate_stress_at_depth(warp, s6r, float(z))[1], float) for z in zs])
    return np.trapz(S, zs, axis=0)


# =====================================================================================
print("=" * 108)
print("TASK 1 -- route 1 (direct A@s6) vs route 2 (int sigma dz), per distinct layup, per station")
print("  FF (VABS) =", FF)
print("  ratio_ply  = plywise-exact-Gauss integral / direct     (quadrature-clean)")
print("  ratio_41   = uniform 41-pt trapz integral / direct      (what dbg_stress_N.py did)")
print("=" * 108)

t_all = time.time()
summary = []
for i in STATIONS:
    B = bundle(i)
    ay = load_station_abd(abd_path(i))
    abd = ay["by_name"]; raw = {L["name"]: L for L in ay["raw"]["layups"]}
    E1 = mat_E1(i)
    st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF)
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
    layups = B["layup_per_elem"]; frac = float(B.get("frac", 0.0))
    L_e = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)

    names = sorted(set(layups))
    print("\n--- station %02d  (r=%.2f)   %d ring elements, %d distinct layups, ref frac=%.1f"
          % (i, i / 50.0, len(rc), len(names), frac))
    print("  %-14s %-11s %5s %4s %5s | %-12s %-12s %-12s | %7s %7s"
          % ("layup", "kind", "h[mm]", "nply", "core", "N11_direct", "N11_int_ply", "N11_int_41",
             "rat_ply", "rat_41"))
    for nm in names:
        es = [e for e in range(len(rc)) if layups[e] == nm]
        e = es[len(es) // 2]                                  # representative element
        kind, cf, h, npl = classify(raw[nm], E1)
        s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        A6 = np.asarray(abd[nm][0])
        Nd = A6[:3, :3] @ s6[:3] + A6[:3, 3:] @ s6[3:6]        # route 1
        w = warp_for(B, nm)
        Sp = N_integral_plywise(w, s6, frac, raw[nm]["plies"])
        Su = N_integral_uniform(w, s6, frac)
        # laminate Voigt [s11,s22,s33,g23,g13,s12] -> N=[N11,N22,N12] uses rows 0,1,5
        Np = np.array([Sp[0], Sp[1], Sp[5]]); Nu = np.array([Su[0], Su[1], Su[5]])
        rp = Np[0] / Nd[0] if abs(Nd[0]) > 1e-9 else np.nan
        ru = Nu[0] / Nd[0] if abs(Nd[0]) > 1e-9 else np.nan
        print("  %-14s %-11s %5.1f %4d %4.0f%% | %+.5e %+.5e %+.5e | %7.4f %7.4f"
              % (nm, kind, 1e3 * h, npl, 100 * cf, Nd[0], Np[0], Nu[0], rp, ru))
        summary.append((i, nm, kind, cf, h, npl, Nd[0], Np[0], Nu[0], rp, ru,
                        Nd[1], Np[1], Nd[2], Np[2], float(np.sum(L_e[es]))))

print("\n--- TASK 1 rollup (N11) ---")
for kind in ("monolithic", "core-thin", "SANDWICH"):
    rows = [s for s in summary if s[2] == kind]
    if not rows:
        continue
    rp = np.array([s[9] for s in rows]); ru = np.array([s[10] for s in rows])
    print("  %-11s n=%3d   plywise-exact ratio: min %.4f  max %.4f  mean %.4f  |  uniform-41 ratio: min %.4f max %.4f mean %.4f"
          % (kind, len(rows), np.nanmin(rp), np.nanmax(rp), np.nanmean(rp),
             np.nanmin(ru), np.nanmax(ru), np.nanmean(ru)))
print("  worst plywise |ratio-1|:")
for s in sorted(summary, key=lambda s: -abs(s[9] - 1))[:6]:
    print("    s%02d %-14s %-11s core=%3.0f%% h=%5.1fmm  ratio_ply=%.4f  ratio_41=%.4f"
          % (s[0], s[1], s[2], 100 * s[3], 1e3 * s[4], s[9], s[10]))

# N22 / N12 too
print("  N22/N12 plywise ratios (all layups):")
r22 = np.array([s[12] / s[11] if abs(s[11]) > 1e-6 else np.nan for s in summary])
r12 = np.array([s[14] / s[13] if abs(s[13]) > 1e-6 else np.nan for s in summary])
print("    N22 ratio  min %.4f max %.4f ;  N12 ratio  min %.4f max %.4f"
      % (np.nanmin(r22), np.nanmax(r22), np.nanmin(r12), np.nanmax(r12)))

# =====================================================================================
print("\n" + "=" * 108)
print("TASK 2 -- 'bug 3':  int A11 ds  (emitted per-element ABD)  vs  EA = C6[0,0] (RM homogenization)")
print("  expected for a thin wall with a FREE hoop (N22=0): EA = int (A11 - A12^2/A22) ds  -> ratio_red ~ 1")
print("  raw int A11 ds / EA is then ~1/(1-nu^2) for isotropic (1.099 at nu=0.3), larger for composites")
print("=" * 108)
print("  %-6s %13s %13s %13s | %8s %8s   %s"
      % ("sta", "EA=C6[0,0]", "int A11 ds", "int A11red ds", "raw/EA", "red/EA", "note"))
for i in STATIONS + [0, 15, 25, 35, 45, 50]:
    B = bundle(i)
    abd = load_station_abd(abd_path(i))["by_name"]
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
    L_e = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)
    layups = B["layup_per_elem"]
    iA = 0.0; iAr = 0.0
    for e in range(len(rc)):
        A = np.asarray(abd[layups[e]][0])[:3, :3]
        iA += A[0, 0] * L_e[e]
        iAr += (A[0, 0] - A[0, 1] ** 2 / A[1, 1]) * L_e[e]
    EA = float(np.asarray(B["Timo"])[0, 0])
    print("  s%02d    %.6e %.6e %.6e | %8.4f %8.4f   perim=%.3f m, %d elems"
          % (i, EA, iA, iAr, iA / EA, iAr / EA, float(L_e.sum()), len(rc)))

# ---- isotropic control: same geometry, one isotropic ply of the same total thickness ----
print("\n  isotropic control (blade_iso: E=30 GPa nu=0.3, same walls; expected raw/EA = 1/(1-nu^2) = 1.0989):")
try:
    import blade_iso as bi
    from emit_abd import emit_station_abd
    for i in [10, 30]:
        Bi = bi.homog_iso(i)
        iy = bi.make_iso_yaml(i)
        ao = os.path.join(os.path.dirname(iy), "abd", os.path.basename(iy).replace(".yaml", "_abd.yaml"))
        if not os.path.exists(ao):
            emit_station_abd(iy, ao, station="iso_s%02d" % i, ref="mid")
        abd = load_station_abd(ao)["by_name"]
        corners = np.asarray(Bi["corners"]); rc = np.asarray(Bi["red_cells"])
        L_e = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)
        lp = Bi["layup_per_elem"]
        iA = sum(np.asarray(abd[lp[e]][0])[0, 0] * L_e[e] for e in range(len(rc)))
        A_ = [np.asarray(abd[lp[e]][0])[:3, :3] for e in range(len(rc))]
        iAr = sum((a[0, 0] - a[0, 1] ** 2 / a[1, 1]) * L_e[e] for e, a in enumerate(A_))
        EA = float(np.asarray(Bi["Timo"])[0, 0])
        print("  iso s%02d  %.6e %.6e %.6e | %8.4f %8.4f" % (i, EA, iA, iAr, iA / EA, iAr / EA))
except Exception as ex:
    print("  iso control skipped:", repr(ex))

# =====================================================================================
print("\n" + "=" * 108)
print("TASK 3 -- cost of route 2 for ONE full station (every ring element), and blade-wide (51 stations)")
print("=" * 108)
i = 10
B = bundle(i)
ay = load_station_abd(abd_path(i)); raw = {L["name"]: L for L in ay["raw"]["layups"]}
st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF)
rc = np.asarray(B["red_cells"]); layups = B["layup_per_elem"]; frac = float(B.get("frac", 0.0))
NE = len(rc)

t0 = time.time()
Wc = {nm: warp_for(B, nm) for nm in sorted(set(layups))}
t_warp = time.time() - t0
print("  per-layup warping setup (%d layups, cached once/station): %.3f s" % (len(Wc), t_warp))

for ng, tag in ((2, "2-pt Gauss/ply"), (4, "4-pt Gauss/ply"), (16, "16-pt Gauss/ply")):
    t0 = time.time(); acc = []
    for e in range(NE):
        s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        acc.append(N_integral_plywise(Wc[layups[e]], s6, frac, raw[layups[e]]["plies"], ng=ng)[0])
    dt = time.time() - t0
    print("  route2 %-16s : %6.2f s for %d elements (%.1f ms/elem)  -> blade 51 sta = %6.1f s"
          % (tag, dt, NE, 1e3 * dt / NE, 51 * (dt + t_warp)))
    acc = np.array(acc)
    if ng == 2:
        n2 = acc
    if ng == 16:
        d = np.abs(acc - n2) / (np.abs(acc).max() + 1e-30)
        print("      2-pt vs 16-pt Gauss: max rel diff %.2e  (2-pt is enough if this is small)" % d.max())

t0 = time.time()
for e in range(NE):
    s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
    A6 = np.asarray(ay["by_name"][layups[e]][0])
    _ = A6[:3, :3] @ s6[:3] + A6[:3, 3:] @ s6[3:6]
print("  route1 direct A@s6      : %6.2f s for %d elements  -> blade 51 sta = %6.1f s"
      % (time.time() - t0, NE, 51 * (time.time() - t0)))
print("\n  (route-1 and route-2 SHARE the _rm_shell_strain step-1 cost; route 2 adds only the plate eval)")
print("\ntotal audit wall time %.1f s" % (time.time() - t_all))
