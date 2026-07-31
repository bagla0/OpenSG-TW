"""recovery_bench.py -- the IN-PLANE stress and 3-D DISPLACEMENT recovery
benchmarks of the OpenSG-RM plate chain, for the four curated Pagano validation
cases (garg caseA + caseC, yu2003 case1 + case2).

This module ADDS the two field families the transverse-stress benchmarks
(examples/garg/pagano_bench.py, examples/yu2003/yu_bench.py) do not plot:

  in-plane   sigma_11, sigma_22, sigma_12 at the sin-peak station x = a/2,
             recovered by Yu-2003 Eq. 66 from the SAME plate solution, compared
             with the exact 3-D solution and a standalone FSDT/CLT chain
  displacement  U1, U2 at x = 0 and U3 at x = a/2 by Yu-2003 Eq. 65,
             U_i = u_i^2d + x3 * phi_i + S(V0 + V1bar + V2)_i, where the plate
             part u^2d = (u0, v0, w, phi1, phi2) is taken FROM ABAQUS (the
             DYMORE role: an external 2-D solver supplies the plate solution,
             OpenSG-RM supplies the through-thickness reconstruction)

STATIONS AND HARMONIC FAMILIES (single harmonic p = pi/a, d/dy = 0)
    sigma_11, sigma_22, sigma_12, sigma_33, w   ~ sin(px)  -> peak at x = a/2
    sigma_13, sigma_23, u, v, phi1, phi2        ~ cos(px)  -> peak at x = 0
So with E6(x) = Es sin(px) the recovery arguments are
    at x = a/2 :  E6 = Es,  E6,1 = 0,       E6,11 = -p^2 Es   (in-plane, U3)
    at x = 0   :  E6 = 0,   E6,1 = p Es,    E6,11 = 0         (U1, U2)
which is why the in-plane recovery NEEDS the second gradient: at the sin peak
the first gradient vanishes and E6,11 carries the whole gradient content.

THE PLATE PART FROM ABAQUS.  The RM-shell decks print U (u1, u2, u3, ur1, ur2,
ur3) along the whole span row NROW0.  `fit_plate_dofs` least-squares fits
    u1(x) = U cos(px) + c1,   u2(x) = V cos(px) + c2,   u3(x) = W sin(px),
    phi1(x) = F1 cos(px),     phi2(x) = F2 cos(px)
with phi1 = UR2 and phi2 = -UR1 (Abaqus rotation vector -> plate rotations:
u = theta x r with r = (0, 0, z) gives du1 = +z ur2 and du2 = -z ur1).  The
constants c1, c2 are DISCARDED: the deck pins u1 (and u2) at one node to remove
the rigid-body modes the simply-supported problem genuinely has, whereas the
exact harmonic solution carries the zero-span-average gauge.  Fitting the
constant out puts both in the same gauge; it changes no strain and no stress.

The fitted (U, V, W, F1, F2) then give the plate strain amplitudes Es = B_E y
that drive the warping, so BOTH the u^2d term and the warping come from the
Abaqus plate solution -- nothing is silently taken from the internal solve.
Pass u2d = "theory" to substitute the internal harmonic solve (the validation
path, and the only option for a case with no Abaqus run).

Run (writes into the ORIGINAL case folders):
    python examples/pagano_recovery/recovery_bench.py                # all four
    python examples/pagano_recovery/recovery_bench.py --case yu2003_case1
    python examples/pagano_recovery/recovery_bench.py --u2d theory
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = the first ancestor holding opensg_jax/
CC = HERE
while not os.path.isdir(os.path.join(CC, "opensg_jax")):
    _up = os.path.dirname(CC)
    if _up == CC:
        raise RuntimeError("opensg_jax repo root not found above " + __file__)
    CC = _up
EX = os.path.join(CC, "examples")
sys.path.insert(0, CC)
sys.path.insert(0, os.path.join(EX, "garg"))
sys.path.insert(0, os.path.join(EX, "yu2003"))

from opensg_jax.fe_jax.msg_rm_plate import (rm_plate_msg,             # noqa: E402
                                            msgrm_strain_at_depth,
                                            msgrm_warping_at_depth)
from opensg_jax.fe_jax.msg_materials import rotated_stiffness_6x6     # noqa: E402
from pagano_exact import ExactCyl                                     # noqa: E402
from statics_fsdt import whitney_k1sq                                 # noqa: E402
import garg_layups                                                    # noqa: E402
import yu_layups                                                      # noqa: E402
from yu_bench import rm_cyl_bend                                      # noqa: E402


# --------------------------------------------------------------------- cases
CASES = {
    "garg_caseA": dict(src="garg", key="caseA", S=10.0, a=1.0, q0=1.0e4,
                       split=False, outdir=os.path.join(EX, "garg", "caseA"),
                       abq=os.path.join(EX, "garg", "caseA", "Abaqus_Plate",
                                        "garg_caseA_S10.dat"),
                       unit="Pa", ulab="m", norm=1.0, qt=1.0, qb=0.0,
                       label="garg caseA [0/90/0], S = a/h = 10"),
    "garg_caseC": dict(src="garg", key="caseC", S=10.0, a=1.0, q0=1.0e4,
                       split=False, outdir=os.path.join(EX, "garg", "caseC"),
                       abq=os.path.join(EX, "garg", "caseC", "Abaqus_Plate",
                                        "garg_caseC_S10.dat"),
                       unit="Pa", ulab="m", norm=1.0, qt=1.0, qb=0.0,
                       label="garg caseC [0/core/0] sandwich, S = a/h = 10"),
    "yu2003_case1": dict(src="yu", key="case1", a=4.0, q0=1.0, split=True,
                         outdir=os.path.join(EX, "yu2003", "case1"),
                         abq=os.path.join(EX, "yu2003", "case1", "Abaqus_Plate",
                                          "yu_case1_RM.dat"),
                         unit="$/p_0$", ulab="in", norm=1.0, qt=0.5, qb=-0.5,
                         label="yu2003 case1 [15/-15], L/h = 4"),
    "yu2003_case2": dict(src="yu", key="case2", a=4.0, q0=1.0, split=True,
                         outdir=os.path.join(EX, "yu2003", "case2"),
                         abq=os.path.join(EX, "yu2003", "case2", "Abaqus_Plate",
                                          "yu_case2_RM.dat"),
                         unit="$/p_0$", ulab="in", norm=1.0, qt=0.5, qb=-0.5,
                         label="yu2003 case2 [30/-30/-30/30], L/h = 4"),
}
# qt / qb: the FACE-PRESSURE fractions of q0 (sigma_33 on the top / bottom
# face) feeding the load ladders of the recovery -- garg is top-loaded,
# the Yu cases carry the split s3 = b3 = p0/2 face load.


def case_setup(name):
    """Assemble everything one case needs: laminate, geometry, exact solver, 8x8.

    Variables
    ---------
    name    a CASES key
    cfg     its config dict (source pipeline, span a, load q0, split-face flag,
            output folder, Abaqus job .dat path, plot units)
    thk     ply thicknesses [m or in], ply 0 at the BOTTOM; for the garg cases
            the layup FRACTIONS are re-scaled to h = a/S, the yu cases are
            already at their paper thickness
    ang, mats, db   ply angles [deg], material names, material database
    h, p    total thickness and the wavenumber pi/a
    ex      the exact solver: top-loaded q0 (garg) or the Yu split face load
            +q0/2 on top and -q0/2 on the bottom
    r       rm_plate_msg result (mid-surface, fraction = 0.5); A6/G2 its blocks
    Returns a dict with all of the above plus the config entries.
    """
    cfg = dict(CASES[name])
    if cfg["src"] == "garg":
        lay = garg_layups.LAYUPS[cfg["key"]]
        db = garg_layups.MATERIAL_DB
        h = cfg["a"] / cfg["S"]
        thk = [t / garg_layups.H * h for t in lay["thick"]]
    else:
        lay = yu_layups.LAYUPS[cfg["key"]]
        db = yu_layups.MATERIAL_DB
        thk = list(lay["thick"])
        h = float(sum(thk))
    ang = list(lay["angles"]); mats = list(lay["mat_names"])
    a = cfg["a"]; q0 = cfg["q0"]; p = np.pi / a
    if cfg["split"]:
        ex = ExactCyl(thk, ang, mats, db, a, q0=0.5 * q0, q_bot=-0.5 * q0)
    else:
        ex = ExactCyl(thk, ang, mats, db, a, q0=q0)
    r = rm_plate_msg(thk, ang, mats, db, fraction=0.5)
    ABDG = np.asarray(r["ABDG"])
    cfg.update(name=name, thk=thk, ang=ang, mats=mats, db=db, h=h, a=a, p=p,
               q0=q0, ex=ex, r=r, ABDG=ABDG, A6=np.asarray(r["A6"]),
               G2=ABDG[6:8, 6:8])
    return cfg


# ------------------------------------------------------- harmonic plate solve
def harmonic_ops(p):
    """The (B_E, B_g) strain operators of the harmonic cylindrical-bending
    kinematics -- the same ones `yu_bench.rm_cyl_bend` builds internally.

    Variables: p = wavenumber [1/len]; y = [U, V, W, F1, F2] the DOF amplitudes
    of u0 = U cos(px), v0 = V cos(px), w = W sin(px), phi1 = F1 cos(px),
    phi2 = F2 cos(px).  B_E (6, 5) maps y to the sin-family plate strain
    amplitudes Es (e11 = -pU, g12 = -pV, k11 = -pF1, k12 = -pF2; e22 = k22 = 0
    for d/dy = 0) and B_g (2, 5) to the cos-family engineering transverse
    shears (2g13 = pW + F1, 2g23 = F2).

    A consistency assertion in `plate_dofs_theory` checks these reproduce
    rm_cyl_bend's own solution, so the duplication cannot drift.
    """
    B_E = np.zeros((6, 5))
    B_E[0, 0] = -p
    B_E[2, 1] = -p
    B_E[3, 3] = -p
    B_E[5, 4] = -p
    B_g = np.zeros((2, 5))
    B_g[0, 2] = p
    B_g[0, 3] = 1.0
    B_g[1, 4] = 1.0
    return B_E, B_g


def plate_dofs_theory(A6, G2, p, q0):
    """The plate DOF amplitudes from the INTERNAL harmonic solve (no FE).

    Variables: A6/G2 = the 6x6 and 2x2 section blocks (MSG or CLT/FSDT);
    p, q0 = wavenumber and total transverse load amplitude; y = [U,V,W,F1,F2];
    Es = B_E y the sin-family strain amplitudes.  Delegates to
    yu_bench.rm_cyl_bend (which also asserts the statics anchors M11 = q0/p^2
    and Q1 = q0/p) and cross-checks the local operators against it.
    """
    y, Es, gs, R6, Q = rm_cyl_bend(A6, G2, p, q0)
    B_E, _ = harmonic_ops(p)
    assert np.allclose(B_E @ y, Es, rtol=1e-10, atol=1e-14), \
        "local harmonic_ops disagrees with rm_cyl_bend"
    return y, Es


# ----------------------------------------------------------- the FSDT/CLT arm
def _qbar(mat, ang, db):
    """The 3x3 plane-stress reduced stiffness Qbar of one ply, Voigt (11,22,12).

    Variables: mat/ang/db = material name, fibre angle [deg], material db;
    C = the rotated 6x6; keep = (11, 22, 12) rows, drop = (33, 23, 13) rows;
    the static condensation Q = C_kk - C_kd C_dd^-1 C_dk imposes sigma_33 =
    sigma_23 = sigma_13 = 0, i.e. the classical plane-stress ply law.
    """
    C = np.asarray(rotated_stiffness_6x6(db[mat]["E"], db[mat]["G"],
                                         db[mat]["nu"], ang))
    keep = np.array([0, 1, 5]); drop = np.array([2, 3, 4])
    return (C[np.ix_(keep, keep)]
            - C[np.ix_(keep, drop)] @ np.linalg.solve(C[np.ix_(drop, drop)],
                                                      C[np.ix_(drop, keep)]))


def clt_blocks(thk, ang, mats, db):
    """The standalone FSDT/CLT section law: classical ABD + Whitney-k shear.

    Variables
    ---------
    zpl        ply interfaces from the MID-surface
    Q          the per-ply plane-stress Qbar (3, 3) from `_qbar`
    A, B, D    the classical laminate matrices, EXACT per-ply integrals
               A = sum Q t, B = sum Q (z2^2-z1^2)/2, D = sum Q (z2^3-z1^3)/3
    A6         (6, 6) [[A, B], [B, D]] in the RM Voigt order
    Gs         (2, 2) int of the ply transverse-shear block [[C55,C45],[C45,C44]]
    k1sq       the Whitney-1973 Eq.-(7) correction from statics_fsdt
    G2         k1sq * Gs -- the single-scalar FSDT shear stiffness (Whitney's
               k2^2 differs slightly; for these cases the in-plane stresses are
               insensitive to it, see the .dat note)
    Returns (A6, G2, Qlist, zpl, k1sq): the section law plus what the pointwise
    FSDT stress recovery needs.
    """
    zpl = np.concatenate([[0.0], np.cumsum(np.asarray(thk, float))])
    zpl -= zpl[-1] / 2
    A = np.zeros((3, 3)); B = np.zeros((3, 3)); D = np.zeros((3, 3))
    Gs = np.zeros((2, 2))
    Qlist = []
    for k, (m, x) in enumerate(zip(mats, ang)):
        Q = _qbar(m, x, db)
        Qlist.append(Q)
        z1, z2 = zpl[k], zpl[k + 1]
        A += Q * (z2 - z1)
        B += Q * (z2 ** 2 - z1 ** 2) / 2.0
        D += Q * (z2 ** 3 - z1 ** 3) / 3.0
        C = np.asarray(rotated_stiffness_6x6(db[m]["E"], db[m]["G"],
                                             db[m]["nu"], x))
        Gs += np.array([[C[4, 4], C[3, 4]], [C[3, 4], C[3, 3]]]) * (z2 - z1)
    A6 = np.block([[A, B], [B, D]])
    k1sq = whitney_k1sq(thk, ang, mats, db)[0]
    return A6, k1sq * Gs, Qlist, zpl, k1sq


def fsdt_inplane(zc, Qlist, zpl, Es):
    """The standalone FSDT in-plane stresses: the CLT staircase Qbar (e0 + z k).

    Variables: zc = query heights from the mid-surface; Qlist/zpl = the per-ply
    Qbar and interfaces from `clt_blocks`; Es = the FSDT plate strain
    amplitudes (first 3 = mid-surface strains e0, last 3 = curvatures k).
    Returns (n, 3) amplitudes in Voigt (11, 22, 12).
    """
    ply = np.clip(np.searchsorted(zpl[1:-1], zc, side="left"), 0,
                  len(Qlist) - 1)
    e0 = Es[:3]; kap = Es[3:]
    return np.array([Qlist[k] @ (e0 + z * kap) for z, k in zip(zc, ply)])


# ------------------------------------------------------ Abaqus node-print I/O
def read_nodeprint(dat_path):
    """Parse every *NODE PRINT table of an Abaqus job .dat.

    Variables
    ---------
    dat_path  the job .dat
    tables    {nset: (n, 1+nval) array} of [node id, values...]; the value
              columns follow `labels`
    labels    the requested identifiers from the table header (U1, U2, U3, UR1,
              UR2, UR3) -- selection downstream is BY LABEL, never by position
    nset      the current "NODE SET <NAME>" context; Abaqus wraps long
              preambles ("...BELONGING TO NODE" / "SET NROW0"), so both the
              one-line and the wrapped form are matched
    rows      data lines "<nid> <values...>"; summary lines (MAXIMUM/MINIMUM/
              NODE) start with a word and are skipped, and any other
              "TABLE IS PRINTED" line closes the context so element tables can
              never be appended to a node table

    Returns {nset: (labels, array)}.
    """
    with open(dat_path, errors="replace") as f:
        lines = f.read().splitlines()
    out = {}
    nset, labels, key, pend = None, None, None, False
    for ln in lines:
        m = re.search(r"NODE SET\s+(\S+)", ln)
        if m:
            nset, labels, key, pend = m.group(1), None, None, False
            continue
        if pend:
            m = re.match(r"\s*SET\s+(\S+)\s*$", ln)
            pend = False
            if m:
                nset, labels, key = m.group(1), None, None
                continue
        if re.search(r"BELONGING TO NODE\s*$", ln):
            pend = True
            continue
        if "TABLE IS PRINTED" in ln:
            nset, labels, key = None, None, None
            continue
        toks = ln.split()
        if nset and not labels and toks and any(
                re.fullmatch(r"UR?\d", t) for t in toks):
            labels = [t for t in toks if re.fullmatch(r"UR?\d", t)]
            key = nset
            out[key] = (labels, [])
            continue
        if key and toks and re.fullmatch(r"\d+", toks[0]):
            vals = [float(toks[0])]
            for t in toks[1:]:
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
            if len(vals) > 1:
                out[key][1].append(vals)
    return {k: (lab, np.array(v)) for k, (lab, v) in out.items() if len(v)}


def fit_plate_dofs(dat_path, a, nel=100):
    """The plate DOF amplitudes y = [U, V, W, F1, F2] fitted from an Abaqus run.

    Variables
    ---------
    dat_path   the RM-shell job .dat, which must print U on the span row NROW0
    a, nel     span and the deck's element count (node i sits at x = i a/nel)
    tab        the parsed NROW0 table; ids -> the node x-coordinates via the
               deck numbering (nid(i, 0) = 2 i + 1, so i = (id - 1) / 2)
    c, s       cos(px) and sin(px) at those stations
    U, V, W    fitted from u1 = U c + c1, u2 = V c + c2 (the constants are the
    F1, F2     deck's rigid-body pins and are DISCARDED -- see the module
               docstring gauge note), u3 = W s, and the rotations
               phi1 = UR2 = F1 c, phi2 = -UR1 = F2 c
    res        the worst relative fit residual over the five fits: a pure single
               harmonic should give ~1e-3 or less (FE discretization only), so a
               large value means the deck or the station map is wrong -- it is
               printed and stored in the .dat header

    Returns (y, res).
    """
    lab, rows = read_nodeprint(dat_path)["NROW0"]
    ofs = rows.shape[1] - len(lab)
    col = {L: rows[:, ofs + i] for i, L in enumerate(lab)}
    x = (rows[:, 0].astype(int) - 1) / 2.0 * (a / nel)
    p = np.pi / a
    c, s = np.cos(p * x), np.sin(p * x)

    def fit(vals, basis, const):
        """Least-squares amplitude of `vals` on `basis` (+ a constant if
        `const`); returns (amplitude, relative residual)."""
        M = np.column_stack([basis, np.ones_like(basis)]) if const \
            else basis[:, None]
        sol, *_ = np.linalg.lstsq(M, vals, rcond=None)
        r = vals - M @ sol
        scale = max(np.max(np.abs(vals)), 1e-300)
        return float(sol[0]), float(np.max(np.abs(r)) / scale)

    U, r1 = fit(col["U1"], c, True)
    V, r2 = fit(col["U2"], c, True)
    W, r3 = fit(col["U3"], s, False)
    F1, r4 = fit(col["UR2"], c, False)       # phi1 = +UR2
    F2, r5 = fit(-col["UR1"], c, False)      # phi2 = -UR1
    # a residual is only meaningful for a component that carries signal: a
    # near-zero field (e.g. the twist rotation of a barely-coupled layup) is
    # FE noise over noise and would dominate the max spuriously
    mags = [np.max(np.abs(col["U1"])), np.max(np.abs(col["U2"])),
            np.max(np.abs(col["U3"])), np.max(np.abs(col["UR2"])),
            np.max(np.abs(col["UR1"]))]
    ref = max(mags)
    res = max(r for r, mg in zip((r1, r2, r3, r4, r5), mags)
              if mg > 1e-6 * ref)
    return np.array([U, V, W, F1, F2]), res


# ------------------------------------------------------------------ benchmarks
def _grid(thk, n_per_ply=81):
    """Through-thickness query grid from the MID-surface, both one-sided limits
    present at every interface (thk = ply thicknesses)."""
    zpl = np.concatenate([[0.0], np.cumsum(np.asarray(thk, float))])
    zpl -= zpl[-1] / 2
    return np.concatenate([np.linspace(zpl[k] + 1e-12, zpl[k + 1] - 1e-12,
                                       n_per_ply)
                           for k in range(len(thk))])


def _relerr(m, e):
    """Relative L2 error [%] of model profile m against exact profile e."""
    return 100 * np.linalg.norm(m - e) / max(np.linalg.norm(e), 1e-300)


def _legend_fig(fig, ax):
    """One shared legend outside the right edge (never blocks a curve)."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.995, 0.5),
               frameon=False, fontsize=10)


def run_inplane(name):
    """In-plane stress recovery at x = a/2: exact vs MSG-RM (Eq. 66) vs FSDT/CLT.

    Variables
    ---------
    cf              the case_setup dict
    zc              the recovery/query grid from the mid-surface
    sig_ex          exact stress amplitudes interpolated onto zc, Voigt
                    [11,22,33,23,13,12] -- columns 0, 1, 5 are the in-plane set
    y_m, Es_m       MSG plate solve DOFs and strain amplitudes
    E6, dE11        the Eq.-66 arguments AT x = a/2: E6 = Es and E6,11 = -p^2 Es
                    (the first gradient vanishes at the sin peak, so without the
                    second gradient the in-plane recovery would lose its whole
                    gradient content)
    s_msg           (n, 3) MSG-RM sigma_11 / sigma_22 / sigma_12 amplitudes
    A6c, G2c, ...   the standalone FSDT/CLT section law and its own plate solve
    s_fsdt          (n, 3) the CLT staircase Qbar (e0 + z k)
    e11/e22/e12     relative L2 errors of each model against exact [%]
    Writes inplane.dat + inplane.png into the case's ORIGINAL folder.
    """
    cf = case_setup(name)
    thk, p, h, a = cf["thk"], cf["p"], cf["h"], cf["a"]
    zc = _grid(thk)
    ze, sige, _, _ = cf["ex"].profile(n_per_layer=81)
    sig_ex = np.column_stack([np.interp(zc, ze, sige[:, j]) for j in range(6)])

    y_m, Es_m = plate_dofs_theory(cf["A6"], cf["G2"], p, cf["q0"])
    z6 = np.zeros(6)
    E6 = Es_m
    dE11 = -p ** 2 * Es_m
    # face-pressure LOAD ladders at the sin peak: q(x) = q_amp sin(px) ->
    # [q, 0, 0, -p^2 q, 0, 0]; they carry the load-driven e33 content sigma_22
    # needs (caseA sigma_22 30% -> 5.6% with them)
    qt6 = np.array([1, 0, 0, -p ** 2, 0, 0]) * (cf["qt"] * cf["q0"])
    qb6 = np.array([1, 0, 0, -p ** 2, 0, 0]) * (cf["qb"] * cf["q0"])
    s_msg = np.empty((len(zc), 3))
    for i, z in enumerate(zc):
        S = msgrm_strain_at_depth(cf["r"], z, E6, z6, z6, dE11, z6, z6,
                                  qt6=qt6, qb6=qb6)[1]
        s_msg[i] = [S[0], S[1], S[5]]

    A6c, G2c, Qlist, zpl, k1sq = clt_blocks(thk, cf["ang"], cf["mats"],
                                            cf["db"])
    _, Es_f = plate_dofs_theory(A6c, G2c, p, cf["q0"])
    s_fsdt = fsdt_inplane(zc, Qlist, zpl, Es_f)

    ex_in = sig_ex[:, [0, 1, 5]]
    # error denominators floored at 1% of ||sigma_11_exact||: for cross-ply
    # laminates sigma_12 (and for near-unidirectional ones sigma_22) is a
    # near-zero field, and a plain relative error on it is noise over noise
    scl = [max(np.linalg.norm(ex_in[:, j]),
               1e-2 * np.linalg.norm(ex_in[:, 0])) for j in range(3)]
    floored = [np.linalg.norm(ex_in[:, j]) < 1e-2 * np.linalg.norm(ex_in[:, 0])
               for j in range(3)]
    err_m = [100 * np.linalg.norm(s_msg[:, j] - ex_in[:, j]) / scl[j]
             for j in range(3)]
    err_f = [100 * np.linalg.norm(s_fsdt[:, j] - ex_in[:, j]) / scl[j]
             for j in range(3)]

    hdr = ["%s -- IN-PLANE stress recovery at the sin-peak station x = a/2"
           % cf["label"],
           "reference surface: MID-SURFACE (fraction = 0.5); plies (bottom->top): "
           + ", ".join("%s(%.4g%s/%g)" % (m, t, cf["ulab"], x)
                       for m, t, x in zip(cf["mats"], thk, cf["ang"])),
           "",
           "MSG-RM: Yu-2003 Eq.-66 recovery with E6 = Es and E6,11 = -p^2 Es",
           "  (at the sin peak E6,1 = 0, so the SECOND gradient carries all of",
           "   the gradient content -- a first-order-only recovery loses it)",
           "  plate DOF amplitudes [U, V, W, F1, F2] = [%s]"
           % ", ".join("%.6g" % v for v in y_m),
           "  strain amplitudes Es = [%s]" % ", ".join("%.6g" % v for v in Es_m),
           "",
           "FSDT/CLT standalone: classical ABD + Whitney-1973 k1^2 = %.6f," % k1sq,
           "  its own harmonic plate solve, stresses = Qbar(z) (e0 + z k).",
           "  (a single scalar k is used for both shear terms; the in-plane",
           "   stresses are insensitive to it for these laminates)",
           "  strain amplitudes Es_fsdt = [%s]"
           % ", ".join("%.6g" % v for v in Es_f),
           "",
           "rel L2 errors vs exact 3-D [%]"
           + (" (floored components:%s -- exact field < 1%% of ||s11||, error "
              "normalized by 1e-2 ||s11|| instead)"
              % "".join(" s" + t for t, f in zip(("11", "22", "12"), floored)
                        if f) if any(floored) else "") + ":",
           "  MSG-RM   s11 %8.4f   s22 %8.4f   s12 %8.4f" % tuple(err_m),
           "  FSDT/CLT s11 %8.4f   s22 %8.4f   s12 %8.4f" % tuple(err_f),
           "",
           "columns: z[%s]  s11_msg s11_exact s11_fsdt  s22_msg s22_exact "
           "s22_fsdt  s12_msg s12_exact s12_fsdt  [%s]"
           % (cf["ulab"], cf["unit"].strip("$/"))]
    np.savetxt(os.path.join(cf["outdir"], "inplane.dat"),
               np.column_stack([zc, s_msg[:, 0], ex_in[:, 0], s_fsdt[:, 0],
                                s_msg[:, 1], ex_in[:, 1], s_fsdt[:, 1],
                                s_msg[:, 2], ex_in[:, 2], s_fsdt[:, 2]]),
               header="\n".join(hdr), fmt="%15.6e")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0))
    comp = [(0, r"\sigma_{11}"), (1, r"\sigma_{22}"), (2, r"\sigma_{12}")]
    for ax, (j, sym) in zip(axes, comp):
        ax.plot(ex_in[:, j], zc / h, "-", color="k", lw=2.0,
                label="exact 3-D (Pagano)" if j == 0 else None)
        ax.plot(s_msg[:, j], zc / h, ":s", color="#ff7f0e", ms=4, mfc="none",
                mew=1.2, lw=1.6, markevery=6,
                label="MSG-RM (Eq. 66)" if j == 0 else None)
        ax.plot(s_fsdt[:, j], zc / h, "--", color="#1f77b4", lw=1.4,
                label="FSDT/CLT\n(Whitney $k_1^2$)" if j == 0 else None)
        ax.set_xlabel(r"$%s$%s  at  $x=a/2$" % (sym, cf["unit"]), fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$z/h$", fontsize=11)
    _legend_fig(fig, axes[0])
    fig.tight_layout()
    fig.savefig(os.path.join(cf["outdir"], "inplane.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  %-14s in-plane  MSG s11 %7.3f%% s22 %7.3f%% s12 %7.3f%%   "
          "(FSDT %6.2f / %6.2f / %6.2f%%)"
          % ((name,) + tuple(err_m) + tuple(err_f)))
    return dict(name=name, err_msg=err_m, err_fsdt=err_f)


def run_disp(name, u2d="abaqus"):
    """3-D displacement recovery (Yu-2003 Eq. 65) with the plate part from Abaqus.

    U1 and U2 are reconstructed at x = 0 (their cos peak) and U3 at x = a/2,
    with the KIRCHHOFF composition (the winner of the controlled caseA sweep,
    S = 4..64 -- see msgrm_warping_at_depth's docstring):
        U1(z) = U - z (p W) + w1(z) ,   U2(z) = V + w2(z)      [at x = 0]
        U3(z) = W + w3(z)                                      [at x = a/2]
    The z-linear term is -z w,1 (w,1 = p W cos(px), amplitude p W at x = 0;
    w,2 = 0 for cylindrical bending), NOT z*phi: the raw warping columns carry
    the mean transverse-shear tilt, which together with the Kirchhoff term
    reproduces the shear-deformable director -- composing with z*phi instead
    double-counts it (caseA U1: 1.41% vs 85% at S = 10).  (w1, w2, w3) =
    msgrm_warping_at_depth with that station's gradient arguments -- E6 = 0,
    E6,1 = p Es at x = 0 (the in-plane warpings are the cos family, driven by
    the FIRST gradient), and E6 = Es, E6,11 = -p^2 Es at x = a/2 (the
    thickness warping is the sin family).

    Variables
    ---------
    u2d          "abaqus" (default) or "theory": where (U, V, W, F1, F2) come
                 from -- the fitted Abaqus span row, or the internal harmonic
                 solve.  Es = B_E y in BOTH cases, so the warping is driven by
                 the same source as the plate part.
    y, Es        the plate DOF and strain amplitudes actually used
    res          the Abaqus harmonic-fit residual (see fit_plate_dofs)
    u1r/u2r/u3r  the recovered amplitude profiles
    u1e/u2e/u3e  the exact amplitudes ex.disp_amp(z) = [U(z), V(z), W(z)]
    avg          <U1_exact> vs the plate u0: the MSG split defines u^2d as the
                 thickness average of the 3-D field, so these must agree; it is
                 printed as an independent check of the gauge handling
    Writes disp_<u2d>.dat + disp_<u2d>.png into the case's ORIGINAL folder.
    """
    cf = case_setup(name)
    thk, p, h, a = cf["thk"], cf["p"], cf["h"], cf["a"]
    zc = _grid(thk)
    B_E, _ = harmonic_ops(p)

    res = float("nan")
    if u2d == "abaqus":
        if not os.path.isfile(cf["abq"]):
            raise FileNotFoundError(
                "no Abaqus job .dat for %s at %s -- run the RM deck first "
                "(or pass --u2d theory)" % (name, cf["abq"]))
        y, res = fit_plate_dofs(cf["abq"], a)
        Es = B_E @ y
    else:
        y, Es = plate_dofs_theory(cf["A6"], cf["G2"], p, cf["q0"])

    z6 = np.zeros(6)
    dE1 = p * Es
    dE11 = -p ** 2 * Es
    # load ladders per station: gradients only at x = 0 (q ~ sin), full at a/2
    qt_e = np.array([0, p, 0, 0, 0, 0]) * (cf["qt"] * cf["q0"])
    qb_e = np.array([0, p, 0, 0, 0, 0]) * (cf["qb"] * cf["q0"])
    qt_m = np.array([1, 0, 0, -p ** 2, 0, 0]) * (cf["qt"] * cf["q0"])
    qb_m = np.array([1, 0, 0, -p ** 2, 0, 0]) * (cf["qb"] * cf["q0"])
    u1r = np.empty_like(zc); u2r = np.empty_like(zc); u3r = np.empty_like(zc)
    for i, z in enumerate(zc):
        w0 = msgrm_warping_at_depth(cf["r"], z, z6, dE1, z6, z6, z6, z6,
                                    qt6=qt_e, qb6=qb_e)
        wm = msgrm_warping_at_depth(cf["r"], z, Es, z6, z6, dE11, z6, z6,
                                    qt6=qt_m, qb6=qb_m)
        u1r[i] = y[0] - z * p * y[2] + w0[0]      # Kirchhoff: -z w,1 = -z p W
        u2r[i] = y[1] + w0[1]                     # w,2 = 0 (d/dy = 0)
        u3r[i] = y[2] + wm[2]                     # + the pressure-compression w3

    ze, _, _, uvw = cf["ex"].profile(n_per_layer=81)
    u1e = np.interp(zc, ze, uvw[:, 0])
    u2e = np.interp(zc, ze, uvw[:, 1])
    u3e = np.interp(zc, ze, uvw[:, 2])
    avg = float(np.trapezoid(u1e, zc) / (zc[-1] - zc[0]))

    e1 = _relerr(u1r, u1e); e3 = _relerr(u3r, u3e)
    # U2 of a cross-ply case is identically ~0: a plain relative error on it is
    # noise/noise.  Floor the denominator at 1e-3 ||U1_exact|| so the reported
    # number stays meaningful (the .dat records when the floor was active).
    u2scale = max(np.linalg.norm(u2e), 1e-3 * np.linalg.norm(u1e))
    e2 = 100 * np.linalg.norm(u2r - u2e) / u2scale
    u2floored = np.linalg.norm(u2e) < 1e-3 * np.linalg.norm(u1e)

    hdr = ["%s -- 3-D DISPLACEMENT recovery, Yu-2003 Eq. 65" % cf["label"],
           "plate part u^2d from: %s%s"
           % ("ABAQUS RM-shell job " + os.path.basename(cf["abq"])
              if u2d == "abaqus" else "the internal harmonic solve (theory)",
              "" if u2d != "abaqus" else
              "  (harmonic fit residual %.2e)" % res),
           "stations: U1, U2 at x = 0 (cos peak); U3 at x = a/2 (sin peak)",
           "",
           "U_1 = U - x3 (p W) + w1 ; U_2 = V + w2 ; U_3 = W + w3   (Kirchhoff",
           "composition -- the raw warping tilt carries the mean shear; z*phi",
           "would double-count it, see msgrm_warping_at_depth)",
           "  plate DOF amplitudes [U, V, W, F1, F2] = [%s]"
           % ", ".join("%.6g" % v for v in y),
           "  (rotations F1/F2 are fitted and recorded but the composition",
           "   needs only U, V, W; Abaqus map phi1 = UR2, phi2 = -UR1; the",
           "   rigid-body constants of the deck's pins are fitted out -- the",
           "   exact solution carries the zero-span-average gauge)",
           "  strain amplitudes Es = [%s]" % ", ".join("%.6g" % v for v in Es),
           "  warping arguments: E6,1 = p Es at x = 0 ; E6 = Es and",
           "  E6,11 = -p^2 Es at x = a/2",
           "",
           "gauge check: <U1_exact> over the thickness = %.6e vs plate u0 = %.6e"
           % (avg, y[0]),
           "",
           "rel L2 errors vs exact 3-D [%%]:  U1 %8.4f   U2 %8.4f%s   U3 %8.4f"
           % (e1, e2, " (U2_exact ~ 0: denominator floored at 1e-3 ||U1||)"
              if u2floored else "", e3),
           "",
           "columns: z[%s]  U1_rec U1_exact  U2_rec U2_exact  U3_rec U3_exact  [%s]"
           % (cf["ulab"], cf["ulab"])]
    np.savetxt(os.path.join(cf["outdir"], "disp_%s.dat" % u2d),
               np.column_stack([zc, u1r, u1e, u2r, u2e, u3r, u3e]),
               header="\n".join(hdr), fmt="%15.6e")

    src = ("OpenSG-RM recovery\n($u^{2d}$ from Abaqus)" if u2d == "abaqus"
           else "OpenSG-RM recovery\n($u^{2d}$ from the plate solve)")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0))
    panels = [(u1e, u1r, r"$U_1$ [%s]  at  $x=0$" % cf["ulab"]),
              (u2e, u2r, r"$U_2$ [%s]  at  $x=0$" % cf["ulab"]),
              (u3e, u3r, r"$U_3$ [%s]  at  $x=a/2$" % cf["ulab"])]
    for ax, (ee, rr, lbl) in zip(axes, panels):
        ax.plot(ee, zc / h, "-", color="k", lw=2.0,
                label="exact 3-D (Pagano)" if ax is axes[0] else None)
        ax.plot(rr, zc / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2,
                lw=1.6, markevery=6, label=src if ax is axes[0] else None)
        ax.set_xlabel(lbl, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$z/h$", fontsize=11)
    _legend_fig(fig, axes[0])
    fig.tight_layout()
    fig.savefig(os.path.join(cf["outdir"], "disp_%s.png" % u2d), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  %-14s disp(%-7s) U1 %8.4f%%  U2 %8.4f%%  U3 %8.4f%%   fit res %.1e"
          % (name, u2d, e1, e2, e3, res))
    return dict(name=name, e1=e1, e2=e2, e3=e3, res=res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="one CASES key (default: all)")
    ap.add_argument("--what", default="both",
                    choices=("inplane", "disp", "both"))
    ap.add_argument("--u2d", default="abaqus", choices=("abaqus", "theory"),
                    help="source of the plate part in the displacement recovery")
    args = ap.parse_args()
    names = [args.case] if args.case else list(CASES)
    print("RM-OpenSG Pagano recovery benchmarks (in-plane + displacement)")
    for nm in names:
        if args.what in ("inplane", "both"):
            run_inplane(nm)
        if args.what in ("disp", "both"):
            run_disp(nm, u2d=args.u2d)
