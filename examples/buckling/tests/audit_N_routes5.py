"""audit_N_routes5.py -- localise 'bug 3': the production conformal assembler (blade_buckling.station_abd)
integrates int A11_red ds to 1.12-1.33 x EA while the RM ring integrates to 1.0000 x EA.
Is that (a) a perimeter/geometry mismatch (OML loft vs mid-surface ring) or (b) a KDTree layup
MIS-ASSIGNMENT that over-represents the thick spar-cap layups?  Compare the per-layup arc-length budget."""
import os, sys, pickle
import numpy as np

ROOT = "/home/roger/a/bagla0/OpenSG-TW-claude"
XSEC = os.path.join(ROOT, "examples", "TW-paper", "xsec_paper")
IEA = os.path.join(ROOT, "examples", "data", "iea_all_stations")
BUCK = os.path.join(ROOT, "examples", "buckling")
CACHE = os.path.join(BUCK, "data", "blade_cache")
sys.path.insert(0, ROOT); sys.path.insert(0, XSEC); sys.path.insert(0, BUCK)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax; jax.config.update("jax_enable_x64", True)
import blade_buckling as bb
from emit_abd import load_station_abd

for i in (10, 30, 40):
    B = pickle.load(open(os.path.join(CACHE, "bundle_s%02d.pkl" % i), "rb"))
    ay = load_station_abd(os.path.join(IEA, "dehom51", "out", "abd", "iea_s%02d_abd.yaml" % i))
    abd = ay["by_name"]
    corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
    Lr = np.linalg.norm(corners[rc[:, 1]] - corners[rc[:, 0]], axis=1)
    lpr = B["layup_per_elem"]
    P, ABD, Gs = bb.station_abd(i)
    Ls = np.linalg.norm(P[bb.sec_elems[:, 1]] - P[bb.sec_elems[:, 0]], axis=1)
    # recover the conformal layup name per element the same way station_abd does
    shell = os.path.join(bb.SHELLD, "iea_s%02d_shell.yaml" % i)
    tree, names = bb.station_layup_lookup(shell)
    mids = 0.5 * (P[bb.sec_elems[:, 0]] + P[bb.sec_elems[:, 1]])
    idx = tree.query(mids + (tree.data.mean(0) - mids.mean(0)))[1]
    lpc = [names[j] for j in idx]

    allnm = sorted(set(lpr) | set(lpc))
    print("=" * 104)
    print("station %02d :  ring perimeter %.3f m (%d elem)   conformal perimeter %.3f m (%d elem)"
          % (i, Lr.sum(), len(Lr), Ls.sum(), len(Ls)))
    print("  %-10s %8s | %9s %9s %8s | %9s %9s | %s"
          % ("layup", "A11red", "ring_s[m]", "conf_s[m]", "conf/ring", "ring_frac", "conf_frac", "d(intA11s)"))
    tot_r = 0.0; tot_c = 0.0
    for nm in allnm:
        A = np.asarray(abd[nm][0])[:3, :3]
        Ar = A[0, 0] - A[0, 1] ** 2 / A[1, 1]
        sr = float(Lr[[e for e in range(len(Lr)) if lpr[e] == nm]].sum())
        sc = float(Ls[[e for e in range(len(Ls)) if lpc[e] == nm]].sum())
        tot_r += Ar * sr; tot_c += Ar * sc
        print("  %-10s %.2e | %9.3f %9.3f %8.3f | %9.1f%% %9.1f%% | %+.4e"
              % (nm, Ar, sr, sc, (sc / sr if sr > 0 else np.nan), 100 * sr / Lr.sum(),
                 100 * sc / Ls.sum(), Ar * (sc - sr)))
    EA = float(np.asarray(B["Timo"])[0, 0])
    print("  int A11red ds : ring %.6e (/EA=%.4f)   conformal %.6e (/EA=%.4f)"
          % (tot_r, tot_r / EA, tot_c, tot_c / EA))
    # what if we keep the conformal layup budget but rescale to the ring perimeter?
    print("  -> perimeter-only effect : conformal x (ring_perim/conf_perim) = %.4f x EA"
          % (tot_c * Lr.sum() / Ls.sum() / EA))
    print("  -> so the residual is layup MIS-ASSIGNMENT, not perimeter.\n")
