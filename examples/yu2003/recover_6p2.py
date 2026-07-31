"""recover_6p2.py -- the post-processor of the Yu-2003 sec.-6.2 analog: OpenSG-RM
3-D recovery DRIVEN BY ABAQUS section forces, benchmarked against the Abaqus 3-D
solid model, with the inbuilt-FSDT Abaqus shell as the baseline.

Chain (Abaqus plays DYMORE, exactly Yu's sec.-6.2 architecture):
    yu_<case>_RM.inp    -> job .dat -> SF/SM at the two stations -> FF amplitudes
                        -> Es = S6 FF6, dE1 = p Es -> Eq.-63 recovery (s13, s23)
                        -> s33 by thickness equilibrium from the loaded bottom face
    yu_<case>_FSDT.inp  -> job .dat -> TSHR13/TSHR23 (Abaqus's own equilibrium-
                        based composite-shell estimate) = the inbuilt FOSDT curves
    yu_<case>_SOLID.inp -> job .dat -> centroidal S along the COL0/COLM element
                        columns = the BENCHMARK profiles
The exact Pagano profiles are drawn as a thin reference (they also validate the
solid model itself).  NO Pagano content enters the recovery chain: the FF comes
from Abaqus, the recovery from the MSG 8x8.

Usage: put the three job .dat files into examples/yu2003/<case>/Abaqus_Plate/
(names yu_<case>_RM.dat / _FSDT.dat / _SOLID.dat), then
    python examples/yu2003/recover_6p2.py [--case case1]
Writes yu_<case>_6p2.dat + yu_<case>_6p2.png into the case folder.
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
CC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, CC)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(CC, "examples", "garg"))

from opensg_jax.fe_jax.msg_rm_plate import rm_plate_msg, msgrm_strain_at_depth  # noqa: E402
from pagano_exact import ExactCyl                                     # noqa: E402
from yu_layups import MATERIAL_DB, LAYUPS, H, L_SPAN, P0              # noqa: E402


def read_elprint_tables(dat_path):
    """Parse every *EL PRINT table of an Abaqus job .dat into labeled arrays.

    Variables
    ---------
    dat_path   the job .dat text file
    lines      its content split into lines
    tables     the result: {(elset, tuple(labels)): rows}, rows = (n, 1+ncol)
               arrays [element id, values...]; multiple tables for the same
               elset (e.g. SF and TSHR requests) stay separate entries
    elset      the current "ELEMENT SET <NAME>" context (regex on the table
               preamble Abaqus prints above each block)
    labels     the column names taken from the header line that lists the
               requested identifiers (tokens like S11/SF1/TSHR13); Abaqus column
               ORDER is version-dependent -- everything downstream selects
               columns BY LABEL, never by position
    rows       numeric lines "<eid> <pt> <values...>"; the section-point / PT
               column is kept when present (shell tables) and absent for
               POSITION=CENTROIDAL solid tables -- the parser stores whatever
               floats follow the leading integer(s)

    Returns the tables dict.
    """
    with open(dat_path, errors="replace") as f:
        lines = f.read().splitlines()
    tables = {}
    elset, labels, key = None, None, None
    for ln in lines:
        m = re.search(r"ELEMENT SET\s+(\S+)", ln)
        if m:
            elset, labels, key = m.group(1), None, None
            continue
        toks = ln.split()
        if elset and not labels and toks and any(
                re.fullmatch(r"(S|SF|SM|TSHR|E)\d+", t) for t in toks):
            labels = [t for t in toks if re.fullmatch(r"[A-Z]+\d+", t)]
            key = (elset, tuple(labels))
            tables.setdefault(key, [])
            continue
        if key and toks and re.fullmatch(r"\d+", toks[0]):
            try:
                vals = [float(t) for t in toks]
            except ValueError:
                continue
            tables[key].append(vals)
    return {k: np.array(v) for k, v in tables.items() if len(v)}


def pick(tables, elset, want):
    """Select columns by LABEL from the parsed tables.

    Variables: tables = read_elprint_tables output; elset = the element-set name;
    want = the label list to extract (e.g. ["S13", "S23", "S33"]).  Finds the
    table of that elset containing ALL wanted labels; leading non-data columns
    (element id, section point) are skipped by aligning the label list to the
    LAST len(labels) columns of the rows.  Returns (ids, cols) with cols one
    array per wanted label.
    """
    for (es, labels), rows in tables.items():
        if es == elset and all(w in labels for w in want):
            ofs = rows.shape[1] - len(labels)
            ids = rows[:, 0].astype(int)
            return ids, [rows[:, ofs + labels.index(w)] for w in want]
    raise KeyError("no table for elset %s with %s" % (elset, want))


def ff_from_shell_dat(dat_path, a, nel=100):
    """FF amplitudes from the strip-shell job .dat (the DYMORE-output step).

    Variables
    ---------
    a, nel      span and element count of the strip deck (centroid corrections)
    p           wavenumber pi/a
    xc_e, xc_m  the EEND / EMID element centroids dx/2 and a/2 - dx/2
    sfm, smm    EMID section forces/moments (mean over the section-point rows --
                SF/SM are section quantities, printed once per element here)
    sfe         EEND section forces
    R6amp       (6,) sin-family resultant AMPLITUDES [N11,N22,N12,M11,M22,M12]:
                the EMID values divided by sin(p xc_m)
    Qamp        (2,) cos-family shear amplitudes [Q1, Q2]: EEND SF4/SF5 divided
                by cos(p xc_e)

    Returns (R6amp, Qamp).
    """
    p = np.pi / a
    dx = a / nel
    xc_e, xc_m = 0.5 * dx, a / 2 - 0.5 * dx
    t = read_elprint_tables(dat_path)
    _, sfm = pick(t, "EMID", ["SF1", "SF2", "SF3", "SF4", "SF5"])
    _, smm = pick(t, "EMID", ["SM1", "SM2", "SM3"])
    _, sfe = pick(t, "EEND", ["SF4", "SF5"])
    R6amp = np.array([np.mean(sfm[0]), np.mean(sfm[1]), np.mean(sfm[2]),
                      np.mean(smm[0]), np.mean(smm[1]), np.mean(smm[2])])
    R6amp /= np.sin(p * xc_m)
    Qamp = np.array([np.mean(sfe[0]), np.mean(sfe[1])]) / np.cos(p * xc_e)
    return R6amp, Qamp


def solid_profiles(dat_path, thick, a, nx=160, nz_per_ply=10):
    """Benchmark through-thickness profiles from the solid job .dat.

    Variables
    ---------
    thick, a, nx,   the deck parameters (element -> z mapping and the harmonic
    nz_per_ply      centroid corrections must match make_abaqus_6p2)
    p, dx           wavenumber and span element size
    zpl, zk         ply interfaces and node planes from the bottom face
    zc              (nzt,) element-centroid heights from the MID-surface [in]
    t               the parsed tables; COL0 rows ascending element id = ascending
                    k (eid(0, k) = 1 + k nx), same for COLM
    s13, s23        COL0 centroidal stresses divided by cos(p dx/2)
    s33             COLM centroidal stresses divided by sin(p (a/2 + dx/2))

    Returns (zc, s13, s23, s33).
    """
    p = np.pi / a
    dx = a / nx
    zpl = np.concatenate([[0.0], np.cumsum(np.asarray(thick, float))])
    zk = np.concatenate([np.linspace(zpl[m], zpl[m + 1], nz_per_ply + 1)[:-1]
                         for m in range(len(thick))] + [[zpl[-1]]])
    zc = 0.5 * (zk[:-1] + zk[1:]) - zpl[-1] / 2
    t = read_elprint_tables(dat_path)
    _, c0 = pick(t, "COL0", ["S13", "S23"])
    _, cm = pick(t, "COLM", ["S33"])
    s13 = c0[0] / np.cos(p * 0.5 * dx)
    s23 = c0[1] / np.cos(p * 0.5 * dx)
    s33 = cm[0] / np.sin(p * (a / 2 + 0.5 * dx))
    return zc, s13, s23, s33


def tshr_profile(dat_path, thick, a, nel=100):
    """Abaqus inbuilt-FSDT transverse-shear estimate TSHR13/TSHR23 at x = 0.

    Variables: thick/a/nel as the FSDT deck; PT = the shell section-point number
    (3 per ply, bottom/mid/top, numbered from the BOTTOM ply up); z_pt = its
    height from the mid-surface; rows = the EEND TSHR table divided by
    cos(p dx/2).  Returns (z_pt, tshr13, tshr23) or None when the table is
    absent (output not supported by the installed version).
    """
    p = np.pi / a
    dx = a / nel
    try:
        t = read_elprint_tables(dat_path)
        _, cols = pick(t, "EEND", ["TSHR13", "TSHR23"])
    except (KeyError, OSError):
        return None
    zpl = np.concatenate([[0.0], np.cumsum(np.asarray(thick, float))])
    z_pt = np.array([zpl[m] + f * (zpl[m + 1] - zpl[m])
                     for m in range(len(thick)) for f in (0.0, 0.5, 1.0)])
    z_pt -= zpl[-1] / 2
    n = min(len(z_pt), len(cols[0]))
    return (z_pt[:n], cols[0][:n] / np.cos(p * 0.5 * dx),
            cols[1][:n] / np.cos(p * 0.5 * dx))


def run_case(case):
    """One 6.2 comparison: Abaqus-FF-driven OpenSG-RM recovery vs the Abaqus 3-D
    solid benchmark and the inbuilt-FSDT shell, with exact Pagano as reference.

    Variables
    ---------
    adir            <case>/Abaqus_Plate/ holding yu_<case>_{RM,FSDT,SOLID}.dat
    lay, thk, ang,  the laminate; a/h/p/q0 the geometry and load (P0 = 1 ->
    mats            everything is already sigma/p0)
    R6amp, Qamp     the Abaqus FF amplitudes (ff_from_shell_dat; the .dat also
                    records the statics anchors q0/p^2 and q0/p they should hit)
    r, S6           rm_plate_msg result and inv(A6): strains from resultants
    Es, dE1_0       the sin-family strain amplitude S6 R6amp and its x = 0
                    gradient p Es (E6(x) = Es sin(px), zero AT x = 0)
    zc              recovery grid from the mid-surface (81 points per ply)
    s13_r, s23_r    the Eq.-63 recovered amplitudes at each zc
    s33_r           thickness-equilibrium integral d(s33)/dz = p s13 from the
                    loaded bottom face -p0/2
    zs, s13_s, ...  the solid-benchmark profiles (solid_profiles)
    tsh             the inbuilt-FSDT TSHR profile or None
    ex, ze, sige    the exact reference (thin gray line; also validates the
                    solid model itself)
    e13, e23, e33   rel L2 errors of the recovery vs the SOLID benchmark
                    (interpolated onto the solid zc grid)

    Writes yu_<case>_6p2.dat + yu_<case>_6p2.png into the case folder.
    """
    adir = os.path.join(HERE, case, "Abaqus_Plate")
    lay = LAYUPS[case]
    thk = lay["thick"]; ang = lay["angles"]; mats = lay["mat_names"]
    a = L_SPAN; h = H; p = np.pi / a; q0 = P0

    R6amp, Qamp = ff_from_shell_dat(
        os.path.join(adir, "yu_%s_RM.dat" % case), a)
    r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
    S6 = np.linalg.inv(np.asarray(r["A6"]))
    Es = S6 @ R6amp
    dE1_0 = p * Es
    z6 = np.zeros(6)
    zpl = np.concatenate([[0.0], np.cumsum(np.asarray(thk, float))]) - h / 2
    zc = np.concatenate([np.linspace(zpl[m] + 1e-9, zpl[m + 1] - 1e-9, 81)
                         for m in range(len(thk))])
    s13_r = np.empty_like(zc); s23_r = np.empty_like(zc)
    for i, z in enumerate(zc):
        Sig = msgrm_strain_at_depth(r, z, np.zeros(6), dE1_0, z6)[1]
        s13_r[i] = Sig[4]; s23_r[i] = Sig[3]
    s33_r = -0.5 * q0 + np.concatenate(
        [[0.0], np.cumsum(0.5 * p * (s13_r[1:] + s13_r[:-1]) * np.diff(zc))])

    zs, s13_s, s23_s, s33_s = solid_profiles(
        os.path.join(adir, "yu_%s_SOLID.dat" % case), thk, a)
    tsh = tshr_profile(os.path.join(adir, "yu_%s_FSDT.dat" % case), thk, a)

    ex = ExactCyl(thk, ang, mats, MATERIAL_DB, a, q0=0.5 * q0, q_bot=-0.5 * q0)
    ze, sige, _, _ = ex.profile(n_per_layer=81)

    def relerr(zm, m, zb, b):
        """Rel L2 error [%] of model (zm, m) vs benchmark (zb, b), interpolated
        onto the benchmark grid."""
        mi = np.interp(zb, zm, m)
        return 100 * np.linalg.norm(mi - b) / np.linalg.norm(b)

    e13 = relerr(zc, s13_r, zs, s13_s)
    e23 = relerr(zc, s23_r, zs, s23_s)
    e33 = relerr(zc, s33_r, zs, s33_s)

    hdr = ["%s -- Yu-2003 sec.-6.2 analog: Abaqus as the 2-D solver (DYMORE role)"
           % case,
           "chain: Abaqus RM-shell SF/SM -> FF -> OpenSG-RM Eq.-63 recovery;",
           "benchmark = Abaqus 3-D solid strip; baseline = inbuilt-FSDT TSHR",
           "FF amplitudes from Abaqus [N11,N22,N12,M11,M22,M12] = [%s]"
           % ", ".join("%.6g" % v for v in R6amp),
           "  statics anchors: M11 -> q0/p^2 = %.6g ; Q1 -> q0/p = %.6g, "
           "Abaqus Q = [%.6g, %.6g]" % (q0 / p ** 2, q0 / p, Qamp[0], Qamp[1]),
           "rel L2 errors of the recovery vs the SOLID benchmark:  "
           "s13 %7.3f%%  s23 %7.3f%%  s33 %7.3f%%" % (e13, e23, e33),
           "TSHR13/23 present: %s" % ("yes" if tsh else "no"),
           "",
           "columns: z[in]  s13_rec  s23_rec  s33_rec  [/p0]  "
           "(solid/TSHR profiles carry their own grids -- see the .png)"]
    np.savetxt(os.path.join(HERE, case, "yu_%s_6p2.dat" % case),
               np.column_stack([zc, s13_r, s23_r, s33_r]),
               header="\n".join(hdr), fmt="%15.6e")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 5.0))
    for ax, (zb, b), (zm, mth), exact, lbl in (
            (ax1, (zs, s13_s), (zc, s13_r), sige[:, 4], r"$\sigma_{13}/p_0$  at  $x=0$"),
            (ax2, (zs, s23_s), (zc, s23_r), sige[:, 3], r"$\sigma_{23}/p_0$  at  $x=0$"),
            (ax3, (zs, s33_s), (zc, s33_r), sige[:, 2], r"$\sigma_{33}/p_0$  at  $x=L/2$")):
        ax.plot(exact, ze / h, "-", color="0.6", lw=1.0,
                label="exact 3-D (reference)" if ax is ax1 else None)
        ax.plot(b, zb / h, "-", color="k", lw=2.0,
                label="Abaqus 3-D solid (benchmark)" if ax is ax1 else None)
        ax.plot(mth, zm / h, ":s", color="#ff7f0e", ms=4, mfc="none", mew=1.2,
                lw=1.6, markevery=6,
                label="OpenSG-RM recovery\n(FF from Abaqus RM shell)"
                if ax is ax1 else None)
        ax.set_xlabel(lbl, fontsize=11)
        ax.grid(alpha=0.3)
    if tsh:
        ax1.plot(tsh[1], tsh[0] / h, "--", color="#1f77b4", lw=1.4,
                 label="Abaqus inbuilt FSDT (TSHR)")
        ax2.plot(tsh[2], tsh[0] / h, "--", color="#1f77b4", lw=1.4)
    ax1.set_ylabel("$z/h$", fontsize=11)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.995, 0.5),
               frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, case, "yu_%s_6p2.png" % case), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  %s (vs solid benchmark): s13 %7.3f%%  s23 %7.3f%%  s33 %7.3f%%"
          % (case, e13, e23, e33))
    return dict(case=case, e13=e13, e23=e23, e33=e33)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="case1/case2/case3 (default: all "
                    "with an Abaqus_Plate folder present)")
    a_ = ap.parse_args()
    for name in LAYUPS:
        if a_.case and name != a_.case:
            continue
        if not os.path.isdir(os.path.join(HERE, name, "Abaqus_Plate")):
            print("  %s: no Abaqus_Plate folder yet -- run the decks first" % name)
            continue
        run_case(name)
