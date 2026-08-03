"""
fig_circle_study.py

Figure: connected multi-section FSM vs per-station-minimum FSM, for the circle
study (prismatic tube / tapered cone, isotropic / -45 anisotropic).

Every number plotted is PARSED FROM THE RUN LOG (default /tmp/circlestudy.log on
msg.ecn.purdue.edu).  Nothing is hardcoded.  If a case is absent from the log it
is simply not drawn, and a note is printed to stderr.

Setup recorded by that run:
    R = 1.0 (prismatic) or 1.0 -> 0.5 (tapered), t = 0.02 m, L = 2.0 m,
    64 strips, unit TOTAL axial force, N11 = -F/(2 pi R(x)) in CLOSED FORM,
    iso = E 200 GPa / nu 0.3;  m45 = single -45 ply, E1 140, E2 10, G12 5 GPa.
Because N11 is closed-form, no dehomogenization enters, so any per-station vs
connected difference is attributable to the FORMULATION alone.

Usage
-----
    python fig_circle_study.py [--log /tmp/circlestudy.log] [--out path.png]
"""

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# log parsing
# ----------------------------------------------------------------------------

# e.g.  "   iso tapered          8  12     2.733177e+08    2.977014e+08   1.0892   112s"
ROW_RE = re.compile(
    r"^\s*(?P<mat>iso|m45)\s+"
    r"(?P<geom>prismatic|tapered)\s+"
    r"(?P<nsec>\d+)\s+"
    r"(?P<M>\d+)\s+"
    r"(?P<per>[-+0-9.eE]+)\s+"
    r"(?P<conn>[-+0-9.eE]+)\s+"
    r"(?P<ratio>[-+0-9.eE]+)"
)


def parse_log(path):
    """Return a list of dicts, one per (mat, geom, nsec, M) row found in the log."""
    with open(path, "r", errors="replace") as fh:
        text = fh.read()

    rows = []
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m is None:
            continue
        d = m.groupdict()
        try:
            rec = dict(
                mat=d["mat"],
                geom=d["geom"],
                nsec=int(d["nsec"]),
                M=int(d["M"]),
                per=float(d["per"]),
                conn=float(d["conn"]),
                ratio=float(d["ratio"]),
            )
        except ValueError:
            continue
        # guard: the log prints the ratio, but recompute it so a mis-parse shows up
        if rec["per"] > 0:
            chk = rec["conn"] / rec["per"]
            if abs(chk - rec["ratio"]) > 5e-4:
                sys.stderr.write(
                    "WARNING: ratio mismatch on row %r: printed %.4f, "
                    "connected/per = %.4f\n" % (d, rec["ratio"], chk)
                )
        rows.append(rec)
    return rows


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------

# Okabe-Ito colourblind-safe palette
C_ISO_LO = "#56B4E9"  # sky blue
C_ISO_HI = "#0072B2"  # blue
C_M45_LO = "#E69F00"  # orange
C_M45_HI = "#D55E00"  # vermillion

GEOM_ORDER = ["prismatic", "tapered"]
GEOM_LABEL = {"prismatic": "prismatic tube", "tapered": "tapered cone"}


def series_style(mat, nsec):
    if mat == "iso":
        col = C_ISO_HI if nsec >= 8 else C_ISO_LO
    else:
        col = C_M45_HI if nsec >= 8 else C_M45_LO
    hatch = "//" if nsec >= 8 else None
    label = r"%s, $n_{\mathrm{sec}}=%d$" % (
        "isotropic" if mat == "iso" else r"$-45^\circ$ ply",
        nsec,
    )
    return col, hatch, label


def make_figure(rows, out_path):
    if not rows:
        raise SystemExit("no data rows parsed from the log -- refusing to draw an empty figure")

    # keep the highest M available for each (mat, geom, nsec): the converged one
    best = {}
    for r in rows:
        key = (r["mat"], r["geom"], r["nsec"])
        if key not in best or r["M"] > best[key]["M"]:
            best[key] = r

    mats = [m for m in ("iso", "m45") if any(k[0] == m for k in best)]
    nsecs = sorted({k[2] for k in best})
    geoms = [g for g in GEOM_ORDER if any(k[1] == g for k in best)]

    # series = (mat, nsec) pairs, drawn side by side inside each geometry group
    series = [(m, n) for m in mats for n in nsecs]
    series = [s for s in series if any(k[0] == s[0] and k[2] == s[1] for k in best)]

    missing = [
        (m, g, n)
        for m in mats
        for g in geoms
        for n in nsecs
        if (m, g, n) not in best
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.3))

    ngrp = len(geoms)
    nser = len(series)
    group_w = 0.78
    bw = group_w / nser
    xg = np.arange(ngrp, dtype=float)

    for i, (mat, nsec) in enumerate(series):
        col, hatch, label = series_style(mat, nsec)
        xs, hs, vals = [], [], []
        for j, geom in enumerate(geoms):
            rec = best.get((mat, geom, nsec))
            if rec is None:
                continue
            x = xg[j] - group_w / 2.0 + bw * (i + 0.5)
            xs.append(x)
            hs.append(rec["ratio"] - 1.0)
            vals.append(rec["ratio"])
        if not xs:
            continue
        ax.bar(
            xs,
            hs,
            width=bw * 0.88,
            bottom=1.0,
            color=col,
            edgecolor=col,
            hatch=hatch,
            linewidth=0.0,
            label=label,
            zorder=3,
        )
        for x, v in zip(xs, vals):
            # cap line so an exactly-1.0000 (zero-height) bar is still visible
            ax.plot(
                [x - bw * 0.44, x + bw * 0.44],
                [v, v],
                color=col,
                lw=2.2,
                solid_capstyle="butt",
                zorder=4,
            )
            ax.annotate(
                "%.4f" % v,
                xy=(x, v),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.6,
                color="0.15",
                rotation=90,
                zorder=5,
            )

    ax.axhline(
        1.0,
        color="0.30",
        lw=1.3,
        ls=(0, (5, 3)),
        zorder=2,
        label="exact prismatic reduction (1.000)",
    )

    ax.set_xticks(xg)
    ax.set_xticklabels([GEOM_LABEL[g] for g in geoms])
    ax.set_ylabel(r"connected $\lambda_1$ / per-station minimum $\lambda_1$")

    vmax = max(r["ratio"] for r in best.values())
    ax.set_ylim(0.985, 1.0 + (vmax - 1.0) * 1.42 + 0.012)
    ax.set_xlim(-0.62, ngrp - 0.38)

    ax.yaxis.grid(True, color="0.88", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("0.4")
    ax.spines["bottom"].set_color("0.4")
    ax.tick_params(axis="both", colors="0.25", length=3)

    # legend: outside the axes, vertical, on the right
    handles, labels = ax.get_legend_handles_labels()
    # redraw bar handles as patches so the hatch shows at legend size
    fixed = []
    for h, l in zip(handles, labels):
        if isinstance(h, plt.Line2D):
            fixed.append((h, l))
        else:
            fixed.append((h, l))
    ax.legend(
        [h for h, _ in fixed],
        [l for _, l in fixed],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        fontsize=9,
        handlelength=1.6,
        borderaxespad=0.0,
    )

    if missing:
        note = "not in log: " + ", ".join(
            "%s %s n=%d" % (m, g, n) for m, g, n in missing
        )
        ax.annotate(
            note,
            xy=(0.0, -0.16),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=7.5,
            color="0.45",
        )

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return best, missing


def main():
    default_out = (
        r"C:\Users\bagla0\ol_fsm_buckling\fig\circle_study_ratio.png"
        if os.name == "nt"
        else os.path.join(os.getcwd(), "circle_study_ratio.png")
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/circlestudy.log")
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()

    if not os.path.isfile(args.log):
        raise SystemExit("log not found: %s" % args.log)

    rows = parse_log(args.log)
    print("parsed %d rows from %s" % (len(rows), args.log))
    for r in rows:
        print(
            "   %-4s %-9s nsec=%d M=%2d  per=%.6e  conn=%.6e  ratio=%.4f"
            % (r["mat"], r["geom"], r["nsec"], r["M"], r["per"], r["conn"], r["ratio"])
        )

    best, missing = make_figure(rows, args.out)
    if missing:
        sys.stderr.write(
            "NOTE: these cases are absent from the log and were OMITTED from the "
            "figure: %s\n" % ", ".join("%s %s nsec=%d" % t for t in missing)
        )
    print("plotted %d bars" % len(best))
    print("wrote %s (%d bytes)" % (args.out, os.path.getsize(args.out)))


if __name__ == "__main__":
    main()
