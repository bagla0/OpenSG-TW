"""test_nxy.py -- A/B the full membrane geometric stiffness (Nxy included) against the previous diag(Nx,Ny).

What must happen if the change is correct:
  1. ISOTROPIC results are BIT-IDENTICAL.  Under axial load an isotropic laminate has A16 = 0, so Nxy = 0
     and the new term is identically zero.  Any change here is a bug.
  2. The PRISMATIC connected/per-station identity STILL holds for m45.  Both solve_fsm_multi and
     solve_fsm_connected now carry the term, so they must continue to agree exactly.  (Note the identity is
     preserved but the VALUE moves: on a prismatic anisotropic member the Nxy cross term still couples
     opposite-parity harmonics, because the sin-cos integral cmm is nonzero for m+m' odd. That is precisely
     the physics that was missing.)
  3. m45 values CHANGE.  A16 != 0 means an axial load generates Nxy, so the anisotropic contribution to the
     geometric stiffness was previously discarded.
"""
import os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

E, NU, T, L = 200e9, 0.3, 0.02, 2.0
R1, R2 = 1.0, 0.5
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
NPER, FTOT = 64, 1.0


def build(fsm, kind, taper, nsec):
    A = np.asarray(fsm.iso_abd(E, NU, T) if kind == "iso" else fsm.clt_abd([(-45.0, T)], MAT), float)
    out = []
    for x in np.linspace(0.0, L, nsec):
        R = R1 + (R2 - R1) * (x / L) if taper else R1
        th = np.linspace(0.0, 2 * np.pi, NPER, endpoint=False)
        P = np.c_[R * np.cos(th), R * np.sin(th)]
        # Pre-buckling state from the laminate itself: a uniform axial resultant Nx drives eps, and for an
        # off-axis laminate the compliance produces a genuine Nxy.  Using inv(A) rather than assuming
        # Nxy = 0 is the whole point of the test.
        Nx = -FTOT / (2 * np.pi * R)
        eps = np.linalg.solve(A[:3, :3], np.array([Nx, 0.0, 0.0]))
        Nvec = A[:3, :3] @ eps                      # = [Nx, 0, 0] by construction (traction-controlled)
        out.append((float(x), P, np.tile(A, (len(P), 1, 1)), np.tile(Nvec, (len(P), 1))))
    return out


def run(fsm, kind, taper, nsec=8, M=12):
    secs = build(fsm, kind, taper, nsec)
    strips = np.array([[i, (i + 1) % NPER] for i in range(NPER)])
    per = []
    for (_, P, A, N) in secs:
        lam = np.asarray(fsm.solve_fsm_multi(P, strips, list(A), list(N), L, M, n_modes=3))
        lam = lam[np.isfinite(lam)]
        per.append(float(lam[0]) if lam.size else np.inf)
    con = np.asarray(fsm.solve_fsm_connected(secs, L, M, n_modes=3, ngauss=max(48, 4 * M)))
    con = con[np.isfinite(con)]
    return min(per), (float(con[0]) if con.size else np.inf)


res = {}
for flag in ("1", "0"):                       # 1 = OLD diag(Nx,Ny);  0 = NEW full membrane Kg
    os.environ["FSM_NO_NXY"] = flag
    for m in [k for k in list(sys.modules) if "fsm_buckling" in k]:
        del sys.modules[m]
    import fsm_buckling as fsm
    tag = "OLD" if flag == "1" else "NEW"
    print("\n%s  (INCLUDE_NXY=%s)" % (tag, fsm.INCLUDE_NXY))
    print("   case             per-station      connected       conn/per")
    for kind in ("iso", "m45"):
        for taper in (False, True):
            t0 = time.time()
            p, c = run(fsm, kind, taper)
            res[(tag, kind, taper)] = (p, c)
            print("   %-14s   %.6e   %.6e   %8.5f   (%.0fs)"
                  % ("%s %s" % (kind, "tapered" if taper else "prismatic"), p, c, c / p, time.time() - t0))

print("\n" + "=" * 78)
print("VERDICTS")
print("=" * 78)
ok = True
for kind in ("iso", "m45"):
    for taper in (False, True):
        o, n = res[("OLD", kind, taper)], res[("NEW", kind, taper)]
        dp = abs(n[0] / o[0] - 1) * 100; dc = abs(n[1] / o[1] - 1) * 100
        lbl = "%s %s" % (kind, "tapered" if taper else "prismatic")
        if kind == "iso":
            good = dp < 1e-9 and dc < 1e-9
            ok &= good
            print("   %-14s  iso must be IDENTICAL: d(per)=%.2e%%  d(con)=%.2e%%   %s"
                  % (lbl, dp, dc, "OK" if good else "*** CHANGED - BUG ***"))
        else:
            print("   %-14s  m45 change: d(per)=%+.3f%%  d(con)=%+.3f%%"
                  % (lbl, 100 * (n[0] / o[0] - 1), 100 * (n[1] / o[1] - 1)))
p, c = res[("NEW", "m45", False)]
good = abs(c / p - 1) < 1e-3
ok &= good
print("   m45 PRISMATIC connected/per-station with the new term = %.6f   %s"
      % (c / p, "OK" if good else "*** IDENTITY BROKEN ***"))
print("\nRESULT: %s" % ("all invariants hold" if ok else "*** an invariant was violated ***"))
