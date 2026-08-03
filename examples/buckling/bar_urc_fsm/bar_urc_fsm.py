"""bar_urc_fsm.py -- FSM local buckling of the BAR-URC 100 m blade, driven by OpenSG-RM (OML reference).

Benchmark to reproduce: eigenvalue ~1.04 with the mode near station 5 in the spar-cap region, from the
3-D-SG solid per-segment sweep under the flapwise 1200 Pa case.

Chain, all native -- no lofted mesh, no nearest-neighbour layup mapping:
    1Dshell_k.yaml (OML contour, GitHub Shell_1DSG)
      -> solve_tw_from_yaml(frac=0.0)      RM homogenization, OML reference
         bundle carries ABD_elems, elements, corners, Timo, L      <- ABD is the SAME one homogenization used,
                                                                      so N and ABD are consistent by construction
      -> dehom under FF from bar_urc-k-t-0.in.glb                  N = A.eps + B.kappa per element
      -> fsm.solve_fsm_multi                                       branched (webbed) contour is supported here
                                                                   (solve_fsm_connected is NOT -- it hardwires
                                                                    a single (i,i+1 mod nn) ring)

Reference conventions, all verified rather than assumed:
  * .glb line 5 = F1 M1 M2 M3, line 6 = F2 F3.  Confirmed three ways: our own writer, dM2/dx1 = F3 to ~1-3%
    across the inboard span, and the FF hardcoded in the existing st15 scripts.
  * frac = 0.0 is the OML reference.  The Shell_1DSG yaml records NO reference key, so it must be passed
    explicitly -- the RM default is "center" and would be wrong for this data.
  * st15 dehom validated vs VABS .SM: median S11 error 0.35% (cap centre) / 1.50% (circumferential).

Known limitation carried into the result: N12 never enters K_G in any of our FSM kernels, so WEB SHEAR
buckling is not captured.  The target spar-cap mode is N11-compression driven, which IS captured, and the
webs still contribute their K_e and so supply the correct rotational restraint at the cap panel edges.
A web-region eigenvalue from this tool must not be read as physical.
"""
import os, sys, time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "buckling"))
sys.path.insert(0, os.path.join(ROOT, "examples", "TW-paper", "xsec_paper"))
sys.path.insert(0, HERE)
import jax
jax.config.update("jax_enable_x64", True)
from opensg_jax.fe_jax import solve_tw_from_yaml
import dehom_rm
import fsm_buckling as fsm
import robust_checks as rc
from vabs_io import read_glb

SHELLD = os.path.join(ROOT, "tests", "data")                 # 1Dshell_0..29.yaml (== GitHub Shell_1DSG)
GLBD = (r"C:\Users\bagla0\OneDrive - purdue.edu\2025_195\20250314_RunallSolidsegments"
        r"\bar_urc_vabs_runs\bar_urc_flap_1200_vabs_npl_2_ar_2p5\temp")
GLBD_SRV = os.path.join(HERE, "glb")                          # server-side copy if the OneDrive path is absent
NSTA = 30
BLADE_LEN = 100.0
DZ = BLADE_LEN / (NSTA - 1)                                   # 3.4483 m station spacing == z_k = k*100/29


def glb_path(k):
    for d in (GLBD, GLBD_SRV):
        p = os.path.join(d, "bar_urc-%d-t-0.in.glb" % k)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no .glb for station %d in %s or %s" % (k, GLBD, GLBD_SRV))


def ff_station(k):
    """FF in VABS order [F1,F2,F3,M1,M2,M3]."""
    g = read_glb(glb_path(k))
    F, M = np.asarray(g["F"], float), np.asarray(g["M"], float)
    return np.array([F[0], F[1], F[2], M[0], M[1], M[2]])


def ring_station(k, M=12, n_modes=4, verbose=True):
    shell = os.path.join(SHELLD, "1Dshell_%d.yaml" % k)
    # The RM bundle (build_rm_bundle) is what the dehom needs -- solve_tw_from_yaml's bundle has no "strip".
    # ref="oml" MUST be explicit: this yaml records no "reference" key, so the default would be "center".
    B = dehom_rm.build_rm_bundle(shell, ref="oml")
    assert float(B["frac"]) == 0.0, "expected OML bundle (frac=0), got frac=%s" % B["frac"]
    # ABD from the frac=0.0 KL bundle = OML-referenced.  NOT the abd yaml build_rm_bundle caches internally,
    # which it writes with ref="mid" -- mixing a mid-ref ABD into an OML contour would silently corrupt B/D
    # (and the EA guard could not catch it, since the A block is reference-independent).
    K = solve_tw_from_yaml(shell, frac=0.0)
    nd = np.asarray(B["corners"], float)                      # (nn,2)
    cells = np.asarray(B["red_cells"], int)
    cells = cells - cells.min()                               # indifferent to 0- vs 1-based labelling
    kcells = np.asarray(K["elements"], int); kcells = kcells - kcells.min()
    assert kcells.shape == cells.shape and (kcells == cells).all(), \
        "element ordering differs between the RM and ABD bundles -- ABD would be assigned to the wrong walls"
    ABD_e = np.asarray(K["ABD_elems"], float)                 # (ns,6,6), OML reference
    ds = np.linalg.norm(nd[cells[:, 1]] - nd[cells[:, 0]], axis=1)

    FF = ff_station(k)
    st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF)
    N_e = np.zeros((len(cells), 3))
    for e in range(len(cells)):
        s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        N_e[e] = ABD_e[e][:3, :3] @ s6[:3] + ABD_e[e][:3, 3:] @ s6[3:6]

    # ---- 1-D ring sanity: solve_fsm_multi divides by the strip length with NO guard ----
    if ds.min() <= 0:
        raise ValueError("station %d has a zero-length strip (min ds=%.3e)" % (k, ds.min()))

    # ---- statics guards ----
    g_eq = rc.section_equilibrium(nd, cells, N_e, FF[4])      # -oint N11 z ds  vs  applied M2
    EA = float(np.asarray(B["Timo"])[0, 0])                   # independent EA from the variational solve
    g_abd = rc.abd_ea_consistency(ABD_e[:, 0, 0], ABD_e[:, 0, 1], ABD_e[:, 1, 1], ds, EA)

    lam = np.asarray(fsm.solve_fsm_multi(nd, cells, list(ABD_e), list(N_e), DZ, M, n_modes=n_modes))
    lam = lam[np.isfinite(lam)]
    lam1 = float(lam[0]) if lam.size else np.inf
    if verbose:
        print("  st%02d  nn=%3d ne=%3d  lam1=%10.4f   equil=%+.3f  abd/EA=%.3f  |N11|max=%.3e N/m"
              % (k, len(nd), len(cells), lam1, g_eq["ratio"], g_abd["ratio"], np.abs(N_e[:, 0]).max()))
    return dict(station=k, nodes=nd, cells=cells, ABD=ABD_e, ds=ds, N=N_e, FF=FF, lam=lam, lam1=lam1,
                guards=[("section_equilibrium", g_eq), ("abd_ea_consistency", g_abd)])


if __name__ == "__main__":
    M = int(os.environ.get("FSM_M", "12"))
    ks = [int(v) for v in sys.argv[1:]] or list(range(3, NSTA))   # st0-2 have no webs
    print("BAR-URC FSM local buckling   (OpenSG-RM, OML ref, flapwise 1200 Pa .glb loads)")
    print("benchmark: lam ~ 1.04 near station 5, spar-cap region;  M=%d, L=%.4f m\n" % (M, DZ))
    out = []
    t0 = time.time()
    for k in ks:
        try:
            out.append(ring_station(k, M=M))
        except Exception as ex:
            print("  st%02d  FAILED  %s: %s" % (k, type(ex).__name__, ex))
    if out:
        gov = min(out, key=lambda r: r["lam1"])
        print("\n  governing: station %d   lambda_1 = %.4f   (%.0f s)" % (gov["station"], gov["lam1"], time.time() - t0))
        srt = sorted(out, key=lambda r: r["lam1"])[:6]
        print("  six lowest:  " + "   ".join("st%02d=%.3f" % (r["station"], r["lam1"]) for r in srt))
        np.savez(os.path.join(HERE, "fsm_scan.npz"),
                 stations=np.array([r["station"] for r in out]),
                 lam1=np.array([r["lam1"] for r in out]))
