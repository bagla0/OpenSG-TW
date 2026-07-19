"""test_robust_checks.py -- prove the pre-flight guards actually FIRE on the known-bad cases and stay quiet
on the known-good ones.  A guard that does not catch the bug we already found is worthless.

Expected:
  section_equilibrium(RM dehom N)  -> PASS   (ratio ~1.0; it satisfies statics)
  section_equilibrium(FE stress)   -> FAIL   (ratio ~0.0; carries none of the beam moment)  <-- the real bug
  fsm_regime(cone)                 -> PASS   (a=0.14 << taper 2.0; per-station is an estimate, 0.5% error)
  fsm_regime(square taper)         -> FAIL   (a~1.0 ~ taper 2.0; per-station is a conservative bound, 18%)
  n_route_consistency(monolithic)  -> PASS   (~0.2%)
  n_route_consistency(sandwich)    -> WARN/FAIL (~18%)
  mesh_sanity(blade)               -> PASS/WARN (closed lofted shell)
"""
import os, sys
import numpy as np
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import robust_checks as rc
import blade_iso as bi
import blade_buckling as bb
import shell_buckling as sb

NSE = bi.NSE; MPER = bb.MPER
ok = True


def expect(name, got, want):
    global ok
    good = got in (want if isinstance(want, (list, tuple)) else [want])
    ok &= good
    print("   %-38s -> %-4s (expected %s) %s" % (name, got, want, "OK" if good else "*** MISMATCH ***"))


print("building blade ...")
bl = bi.build()
nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
f = bb.traction_load(nodes, quads)
FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root)
Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)
Nr = bi.rm_N(bl, FF)

print("\n1) section_equilibrium  (the check that identified which stress field is physical)")
for i in [15, 25]:
    p = min(i * MPER, bl["NS"] - 2)
    sl = slice(p * NSE, (p + 1) * NSE)
    crm = rc.section_equilibrium(bl["Pk"][i], bb.sec_elems, Nr[sl], FF[i][4])
    cfe = rc.section_equilibrium(bl["Pk"][i], bb.sec_elems, Nf[sl], FF[i][4])
    print("   sta %2d: RM ratio=%+.3f  FE ratio=%+.3f" % (i, crm["ratio"], cfe["ratio"]))
    expect("sta%d RM dehom N (physical)" % i, crm["verdict"], rc.PASS)
    expect("sta%d FE membrane N (broken)" % i, cfe["verdict"], rc.FAIL)

print("\n2) fsm_regime  (does it separate shell-type from plate-type buckling?)")
cone = rc.fsm_regime(a_crit=0.141, station_spacing=1.0, width_at_station=1.0)      # sqrt(R t), 3 secs over L=2
sq = rc.fsm_regime(a_crit=1.0, station_spacing=1.0, width_at_station=1.0)          # plate buckle ~ wall width
print("   cone a/spacing=%.3f  square a/spacing=%.3f" % (cone["a_over_spacing"], sq["a_over_spacing"]))
expect("cone (local buckle)", cone["verdict"], rc.PASS)
expect("square taper (buckle spans taper)", sq["verdict"], [rc.WARN, rc.FAIL])

print("\n3) n_route_consistency  (direct A*s6 vs integral of the dehom 3-D stress)")
mono = rc.n_route_consistency(np.array([1.0, 2.0, -3.0]), np.array([1.002, 2.004, -3.006]))
sand = rc.n_route_consistency(np.array([1.0, 2.0, -3.0]), np.array([0.818, 1.636, -2.454]))
print("   monolithic max_dev=%.4f   sandwich max_dev=%.4f" % (mono["max_dev"], sand["max_dev"]))
expect("monolithic walls agree", mono["verdict"], rc.PASS)
expect("sandwich walls diverge", sand["verdict"], [rc.WARN, rc.FAIL])

print("\n4) mesh_sanity (blade lofted shell)")
ms = rc.mesh_sanity(nodes, quads)
print("   %s" % {k: v for k, v in ms.items() if k != "note"})

print("\n" + rc.report([
    ("section_equil(RM)", rc.section_equilibrium(bl["Pk"][25], bb.sec_elems,
                                                 Nr[25 * MPER * NSE:(25 * MPER + 1) * NSE], FF[25][4])),
    ("section_equil(FE)", rc.section_equilibrium(bl["Pk"][25], bb.sec_elems,
                                                 Nf[25 * MPER * NSE:(25 * MPER + 1) * NSE], FF[25][4])),
    ("fsm_regime(cone)", cone), ("fsm_regime(square)", sq),
    ("n_route(sandwich)", sand), ("mesh_sanity", ms),
], title="blade pre-flight (mixed good/bad on purpose)"))

print("\nALL GUARD EXPECTATIONS %s" % ("MET" if ok else "**NOT** met"))
sys.exit(0 if ok else 1)
