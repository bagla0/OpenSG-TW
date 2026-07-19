"""blade_fsm.py -- ROBUST per-station FSM local buckling of a realistic blade, driven entirely by NATIVE
cross-section data.

WHY THIS EXISTS.  The earlier blade path built a fixed-topology "conformal" section (NSE~135 elements) so a
3-D shell mesh could be lofted, then mapped each station's layups onto it with a nearest-midpoint KDTree.
That mapping is the single largest error source for a REALISTIC composite blade:

  * the conformal contour and the station's own ring are DIFFERENT curves, so nearest-neighbour projection
    distorts how much arc length each layup receives;
  * the stiff spar cap is over-assigned and the soft panel starved, so the reduced-axial identity
    oint (A11 - A12^2/A22) ds / EA  drifts 1.03 -> 1.33 growing outboard;
  * that mis-scales the pre-buckling N -- and hence the buckling load -- by 3-33%;
  * sub-sampling along the element does NOT fix it (measured: 1.035->1.111, 1.326->1.371), because the
    defect is geometric, not a resolution problem.

The FSM needs NO lofted mesh.  Per station it needs exactly three things, and all three already exist on
the station's own ring, where the dehomogenization is evaluated anyway:
    contour (ring nodes/elements)  +  per-wall ABD (native layup map)  +  pre-buckling N (dehom on the ring)
Using them directly removes the mapping entirely -- there is nothing left to mis-assign.

Each station is validated by statics before its buckling load is believed (robust_checks):
    section_equilibrium  : -oint N11 z ds  must equal the applied beam moment  (catches a wrong N)
    abd_ea_consistency   : oint (A11 - A12^2/A22) ds  must equal EA            (catches a wrong ABD)
    fsm_regime           : is the buckle short vs the station spacing?         (estimate vs lower bound)
"""
import os, sys
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fsm_buckling as fsm
import robust_checks as rc
import blade_buckling as bb
import dehom_rm
from emit_abd import load_station_abd


def ring_station(i, FF_i, n_modes=4, M=12):
    """Native per-station FSM inputs + solve.  Returns dict with contour, ABD, N, eigenvalues and guards.

    i     station index (0..NSTA-1)
    FF_i  applied beam force at this station, VABS order [F1,F2,F3,M1,M2,M3]
    """
    shell = os.path.join(bb.SHELLD, "iea_s%02d_shell.yaml" % i)
    d = yaml.safe_load(open(shell))
    nd = np.array([bb._row(n)[:2] for n in d["nodes"]], float)
    cells = np.array([[int(x) for x in bb._row(e)] for e in d["elements"]]); cells -= cells.min()
    name_of = {}
    for grp in d["sets"]["element"]:
        for lab in grp["labels"]:
            name_of[int(lab) - 1] = grp["name"]
    names = [name_of.get(k, d["sections"][0]["elementSet"]) for k in range(len(cells))]
    ay = load_station_abd(os.path.join(bb.ABDD, "iea_s%02d_abd.yaml" % i))["by_name"]

    # --- native per-element ABD (no mapping: the yaml says which layup each element IS) ---
    ABD_e = np.array([np.asarray(ay[nm][0], float) for nm in names])
    ds = np.linalg.norm(nd[cells[:, 1]] - nd[cells[:, 0]], axis=1)

    # --- pre-buckling N straight from the dehom, evaluated AT each ring element ---
    B = bb.homogenize_station(i)
    st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF_i)
    N_e = np.zeros((len(cells), 3))
    for e in range(len(cells)):
        s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
        N_e[e] = ABD_e[e][:3, :3] @ s6[:3] + ABD_e[e][:3, 3:] @ s6[3:6]

    # --- guards (statics, no external reference needed) ---
    g_eq = rc.section_equilibrium(nd, cells, N_e, FF_i[4])
    # EA must come from the INDEPENDENT homogenized 6x6 (key "Timo"), NOT from the ring sum -- comparing the
    # ring to itself makes abd_ea_consistency self-referential and it would report 1.000 unconditionally.
    _T = B.get("Timo", None)
    if _T is None:
        raise KeyError("bundle has no homogenized 6x6 ('Timo'); abd_ea_consistency needs an independent EA")
    EA = float(np.asarray(_T)[0, 0])
    g_abd = rc.abd_ea_consistency(ABD_e[:, 0, 0], ABD_e[:, 0, 1], ABD_e[:, 1, 1], ds, EA)

    # --- FSM buckling on the native contour ---
    L = float(bb.BLADE_LEN) / (bb.NSTA - 1)          # station spacing = the span the buckle may occupy
    lam = np.asarray(fsm.solve_fsm_multi(nd, cells, list(ABD_e), list(N_e), L, M, n_modes=n_modes))
    lam = lam[np.isfinite(lam)]
    a_crit = L / max(1, M)                            # shortest resolved half-wave
    g_reg = rc.fsm_regime(a_crit=a_crit, station_spacing=L, width_at_station=float(ds.max()))

    return dict(station=i, nodes=nd, cells=cells, names=names, ABD=ABD_e, ds=ds, N=N_e,
                lam=lam, lam1=float(lam[0]) if lam.size else np.inf,
                guards=[("section_equilibrium", g_eq), ("abd_ea_consistency", g_abd), ("fsm_regime", g_reg)])


def blade_scan(stations, FF, verbose=True):
    """Run every station, report per-station buckling factor + guard verdicts, return the governing station."""
    out = []
    for i in stations:
        r = ring_station(i, FF[i])
        worst = max((g[1]["verdict"] for g in r["guards"]),
                    key=lambda v: {rc.PASS: 0, rc.WARN: 1, rc.FAIL: 2}[v])
        r["worst"] = worst; out.append(r)
        if verbose:
            print("  s%02d  n_elem=%3d  lam1=%.4e  equil=%+.3f  abd/EA=%.3f  [%s]"
                  % (i, len(r["cells"]), r["lam1"], r["guards"][0][1]["ratio"],
                     r["guards"][1][1]["ratio"], worst))
    trust = [r for r in out if r["worst"] != rc.FAIL]
    gov = min(trust or out, key=lambda r: r["lam1"])
    if verbose:
        print("\n  governing station s%02d  lambda_1 = %.4e   (%d/%d stations passed the guards)"
              % (gov["station"], gov["lam1"], len(trust), len(out)))
    return out, gov
