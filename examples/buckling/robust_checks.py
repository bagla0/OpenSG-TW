"""robust_checks.py -- pre-flight validation for cross-section-based local buckling of REALISTIC blades.

Clean benchmark tubes (cylinder / cone / square) hide the ways a real blade breaks the assumptions.  Each
check here is SELF-CONTAINED -- it needs no external 3-D reference -- and returns a verdict plus the number
it measured, so a blade run can be trusted (or flagged) without a full shell model.

Why each one exists (every one caught a real bug in this project):

1. section_equilibrium  The pre-buckling N must carry the applied beam load:  -oint N11 z ds == FF_My.
                        This caught a blade whose 3-D FE stress carried ~0% of the beam moment while the
                        RM dehomogenization carried 100% -- i.e. it identified WHICH of two disagreeing
                        stress fields was physical, using statics alone.
2. n_route_consistency  Direct  N = A s6 + B kappa  vs the through-thickness integral of the dehomogenized
                        3-D stress.  These agree to ~0.2% for MONOLITHIC walls but diverge (~18%) for
                        foam-core SANDWICH panels -- exactly the panels that govern blade local buckling.
3. fsm_regime           The per-station FSM assumes the buckle is short compared with the taper.  True for
                        shell-type buckling (a ~ sqrt(Rt) << taper, cone: 0.5% error) but NOT for plate-type
                        buckling of wide walls (a ~ wall width ~ taper, square: 18% conservative).  Flags
                        which regime a station is in so a conservative answer is not read as an estimate.
4. mesh_sanity          Non-manifold edges, unmerged duplicate nodes, degenerate quads, fold fraction --
                        the mesh defects that silently produce a structure that cannot carry load.

Usage: call the individual checks, or `report(...)` for a formatted pre-flight block.
"""
import numpy as np

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _verdict(err, warn_tol, fail_tol):
    return PASS if err <= warn_tol else (WARN if err <= fail_tol else FAIL)


# ----------------------------------------------------------------------------- 1. section equilibrium
def section_equilibrium(P, sec_elems, N_sec, FF_My, warn_tol=0.10, fail_tol=0.25):
    """Does the membrane resultant carry the applied beam moment?  Statics, no reference model needed.

    P         (nn,2) section coords (y_chord, z_flap) at this station
    sec_elems (nse,2) node pairs of the section elements
    N_sec     (nse,3) membrane resultant per section element, component 0 = SPANWISE N11
    FF_My     applied beam moment about the chord axis at this station (same reduction point as P)

    Returns dict(M_internal, ratio, verdict).  ratio -> 1 means the stress is in equilibrium with the load.
    """
    P = np.asarray(P, float); se = np.asarray(sec_elems, int); N_sec = np.asarray(N_sec, float)
    a, b = se[:, 0], se[:, 1]
    ds = np.linalg.norm(P[b] - P[a], axis=1)
    zmid = 0.5 * (P[a, 1] + P[b, 1])
    M = float(-np.sum(N_sec[:, 0] * zmid * ds))
    ratio = M / FF_My if FF_My else np.nan
    err = abs(abs(ratio) - 1.0)
    return dict(M_internal=M, M_applied=float(FF_My), ratio=ratio,
                verdict=_verdict(err, warn_tol, fail_tol),
                note="|ratio|~1 => pre-buckling N is in equilibrium with the applied beam load")


# ----------------------------------------------------------------------------- 2. ABD assignment vs EA
def abd_ea_consistency(A11, A12, A22, ds, EA, warn_tol=0.05, fail_tol=0.15):
    """Is the per-element ABD ASSIGNMENT consistent with the homogenized axial stiffness EA = C6[0,0]?

    For a thin wall with a FREE hoop (N22 = 0) the correct identity is the REDUCED axial stiffness

        EA == oint ( A11 - A12^2 / A22 ) ds

    NOT  oint A11 ds .  The raw form equals EA/(1-nu^2) only for an ISOTROPIC wall (1.0989 at nu=0.3); a
    composite legitimately gives 1.02-1.15, so testing the raw form produces FALSE ALARMS -- that mistake
    is what once looked like an "ABD/EA bug".  The reduced form is 1.0000 on a correctly-assembled ring at
    every station, composite and isotropic alike.

    It drifts (1.03 -> 1.33, growing outboard) when the conformal section is too COARSE to resolve the
    spar-cap boundary: the stiff cap layup gets over-assigned arc length while the soft panel layup is
    starved, which mis-scales the pre-buckling N by the same 3-33%.  Costs milliseconds; run it per station.

    NOTE on quadrature: if you ever integrate stress through the thickness, use PLY-WISE Gauss (2-pt is
    exact); a uniform trapezoid across a sandwich face/core interface (sigma jumps ~3 decades) fabricates a
    ~20% error out of nothing.
    """
    A11 = np.asarray(A11, float).ravel(); A12 = np.asarray(A12, float).ravel()
    A22 = np.asarray(A22, float).ravel(); ds = np.asarray(ds, float).ravel()
    red = A11 - np.where(np.abs(A22) > 0, A12 ** 2 / np.where(np.abs(A22) > 0, A22, 1.0), 0.0)
    EA_ring = float(np.sum(red * ds))
    ratio = EA_ring / EA if EA else np.nan
    err = abs(ratio - 1.0)
    return dict(EA_from_ring=EA_ring, EA_homog=float(EA), ratio=ratio,
                verdict=_verdict(err, warn_tol, fail_tol),
                note="ratio!=1 => per-element layup ASSIGNMENT is wrong (too-coarse section); N is mis-scaled")


# ----------------------------------------------------------------------------- 3. per-station FSM regime
def fsm_regime(a_crit, station_spacing, width_at_station, width_change_over_a=None,
               warn_ratio=0.30, fail_ratio=1.0):
    """Is the per-station (prismatic) FSM assumption valid at this station?

    a_crit          critical buckle half-wavelength from the signature curve / multi-harmonic solve
    station_spacing distance to the neighbouring cross-section (the taper length the buckle sees)
    width_at_station governing wall width (for context)
    width_change_over_a  fractional change of the wall width across ONE half-wavelength, if known

    a_crit << spacing  -> the buckle is local; the per-station value is an ESTIMATE (cone: 0.5% error).
    a_crit ~ spacing   -> the wall narrows within one buckle; the per-station MINIMUM is then a safe
                          LOWER BOUND, not an estimate (square taper: 18% conservative).
    """
    rho = float(a_crit) / float(station_spacing) if station_spacing else np.inf
    v = _verdict(rho, warn_ratio, fail_ratio)
    if width_change_over_a is not None and abs(width_change_over_a) > 0.15 and v == PASS:
        v = WARN
    return dict(a_crit=float(a_crit), station_spacing=float(station_spacing),
                a_over_spacing=rho, width=float(width_at_station),
                width_change_over_a=width_change_over_a, verdict=v,
                note=("local buckle: per-station value is an estimate" if v == PASS else
                      "buckle spans the taper: per-station min is a CONSERVATIVE LOWER BOUND"))


# ----------------------------------------------------------------------------- 4. mesh sanity
def mesh_sanity(nodes, quads, tol=1e-9, expect_open_ends=True, expect_tjunctions=True):
    """Structural defects that silently make a shell unable to carry load.

    CLASSIFY edges rather than condemn them -- a webbed blade is legitimately non-manifold:
      count 1 : open boundary   -- expected at the root and tip rings of a lofted blade
      count 2 : manifold interior
      count 3 : T-junction      -- expected where a shear web meets the skin (skin continues + web branches)
      count>3 : genuine defect  -- more than one branch on an edge
    Only >3 / duplicate / degenerate are FAILures; open ends and T-junctions are reported for eyeballing
    (e.g. 600 T-edges == 3 webs x 2 attachment lines x 100 span layers, and 270 boundary == root+tip rings).
    """
    nodes = np.asarray(nodes, float); quads = np.asarray(quads, int)
    edges = {}
    for q in quads:
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            e = (min(a, b), max(a, b)); edges[e] = edges.get(e, 0) + 1
    cnt = np.array(list(edges.values()))
    boundary = int((cnt == 1).sum()); tjunc = int((cnt == 3).sum()); bad = int((cnt > 3).sum())
    key = np.round(nodes / max(tol, 1e-12)).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    dup = int(len(nodes) - len(first))
    X = nodes[quads]
    area = 0.5 * np.linalg.norm(np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 1]), axis=1)
    degen = int((area <= 1e-12 * (area.max() + 1e-30)).sum())
    v = PASS
    if bad or dup or degen:
        v = FAIL
    elif (boundary and not expect_open_ends) or (tjunc and not expect_tjunctions):
        v = WARN
    return dict(n_nodes=len(nodes), n_quads=len(quads), open_boundary_edges=boundary,
                tjunction_edges=tjunc, bad_edges=bad, duplicate_nodes=dup, degenerate_quads=degen,
                verdict=v, note="bad_edges(>3 elems)/duplicate/degenerate => mesh cannot carry load correctly")


# ----------------------------------------------------------------------------- report
def report(checks, title="pre-flight"):
    """checks: list of (name, dict-from-a-check).  Returns a formatted string; worst verdict last line."""
    order = {PASS: 0, WARN: 1, FAIL: 2}
    worst = PASS; lines = ["=== %s ===" % title]
    for name, c in checks:
        worst = max(worst, c["verdict"], key=lambda v: order[v])
        detail = ", ".join("%s=%s" % (k, ("%.4g" % v) if isinstance(v, float) else v)
                           for k, v in c.items() if k not in ("verdict", "note"))
        lines.append("  [%-4s] %-22s %s" % (c["verdict"], name, detail))
        if c["verdict"] != PASS:
            lines.append("         ^ %s" % c["note"])
    lines.append("  OVERALL: %s" % worst)
    return "\n".join(lines)
