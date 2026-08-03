"""circle_study.py -- PRISMATIC tube and TAPERED cone, isotropic and anisotropic, per-station vs CONNECTED.

This is the controlled counterpart to the BAR-URC blade run.  The pre-buckling state is known in closed form
(uniform axial compression, N11 = -F/(2 pi R(x))), so nothing here depends on a dehomogenization -- any
difference between the per-station minimum and the connected solve is attributable to the FORMULATION alone.

Cases:
   prismatic  R = 1.0 constant       iso  and  m45 (single -45 ply)
   tapered    R = 1.0 -> 0.5         iso  and  m45
For each: per-station minimum over nsec prismatic sections, and the connected multi-section solve.

Expectation being tested.  On a PRISMATIC member the harmonics decouple, so connected MUST equal
per-station exactly (ratio 1.000) -- that is the sanity check.  On a TAPER the orthogonality breaks, the
harmonics couple, and the connected value should differ; the per-station minimum picks the weakest single
section, which is a section the buckle cannot actually occupy in isolation.
"""
import os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fsm_buckling as fsm

E, NU, T, L = 200e9, 0.3, 0.02, 2.0
R1, R2 = 1.0, 0.5
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)
NPER = 64
FTOT = 1.0                      # unit total axial force; lam is then the critical total force


def ring(R, n=NPER):
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.c_[R * np.cos(th), R * np.sin(th)]


def abd_of(kind):
    if kind == "iso":
        return np.asarray(fsm.iso_abd(E, NU, T), float)
    return np.asarray(fsm.clt_abd([(-45.0, T)], MAT), float)


def sections(kind, taper, nsec):
    """(x, nodes, ABD per strip, N per strip) with N11 = -F/(2 pi R(x)) -- exact for uniform axial load."""
    A = abd_of(kind)
    out = []
    for x in np.linspace(0.0, L, nsec):
        R = R1 + (R2 - R1) * (x / L) if taper else R1
        P = ring(R)
        N = np.tile(np.array([-FTOT / (2 * np.pi * R), 0.0, 0.0]), (len(P), 1))
        out.append((float(x), P, np.tile(A, (len(P), 1, 1)), N))
    return out


def run(kind, taper, nsec=8, M=12):
    secs = sections(kind, taper, nsec)
    per = []
    for (_, P, A, N) in secs:                       # each section as an isolated prismatic member
        lam = np.asarray(fsm.solve_fsm_multi(P, np.array([[i, (i + 1) % len(P)] for i in range(len(P))]),
                                             list(A), list(N), L, M, n_modes=3))
        lam = lam[np.isfinite(lam)]
        per.append(float(lam[0]) if lam.size else np.inf)
    t0 = time.time()
    con = np.asarray(fsm.solve_fsm_connected(secs, L, M, n_modes=3, ngauss=max(48, 4 * M)))
    con = con[np.isfinite(con)]
    c1 = float(con[0]) if con.size else np.inf
    return np.array(per), c1, time.time() - t0


print("PRISMATIC TUBE and TAPERED CONE -- per-station minimum vs CONNECTED multi-section FSM")
print("   R %.2f%s,  t=%.3f m,  L=%.1f m,  %d strips,  unit total axial force\n"
      % (R1, "" if False else " -> %.2f (taper)" % R2, T, L, NPER))
print("   case                nsec   M    per-station min      connected      connected/per   time")
for kind in ("iso", "m45"):
    for taper in (False, True):
        for nsec in (4, 8):
            per, con, dt = run(kind, taper, nsec=nsec)
            tag = "%s %s" % (kind, "tapered" if taper else "prismatic")
            note = ""
            if not taper:
                note = "  <- must be 1.000 (harmonics decouple)" if abs(con / per.min() - 1) < 1e-3 \
                    else "  *** PRISMATIC MISMATCH ***"
            print("   %-18s %3d  %2d   %14.6e  %14.6e   %10.4f   %4.0fs%s"
                  % (tag, nsec, 12, per.min(), con, con / per.min(), dt, note))
    print()

# harmonic convergence on the tapered cases (the prismatic ones are exact at any M >= m*)
print("   harmonic convergence, tapered, nsec=8:")
print("      case     M=6         M=12        M=18        M=24")
for kind in ("iso", "m45"):
    row = []
    for M in (6, 12, 18, 24):
        _, con, _ = run(kind, True, nsec=8, M=M)
        row.append(con)
    print("      %-6s " % kind + "  ".join("%.4e" % v for v in row))
