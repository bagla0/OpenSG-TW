"""RM-regime guard: per station, decide whether the RM 1D-shell homogenization can REPLACE the 2D-solid,
or whether a solid check is warranted -- from the thin-wall metric (t_max/h) and soft-core sandwich content.

Grounded in: the thin-wall ABD reduction is O(h/c)-accurate and degrades quadratically; transverse shear
(GA2/GA3) is the first 6x6 term to fail; soft-core sandwich is the theory's weakest case (shear path lives
in the low-G core). Thresholds calibrated to OUR measured data:
  * IEA-22  t_max/h 2-9%  -> RM within ~7% root-to-tip, +24% only at the very tip (t/h peak 9.2%).
  * mh104   thick airfoil -> GA3 +133% (transverse shear), even webless -> wall-collapse limit.
So: t/h < 5% trust all terms; 5-10% trust EA/EI/GJ but watch GA2/GA3; >=10% keep solid for shear.
A soft core that is a large fraction of the SECTION height (t_core/h) escalates the same way, since its
shear compliance then matters at section level.

Usage:  python rm_regime_guard.py [case]      (default case=iea22; reads <case>_stations.dat + shell_<case>_<tag>.yaml)
Reusable for ANY blade: point it at that blade's stations.dat + shell YAMLs.
"""
import os, sys, glob
import yaml

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
VAL = os.path.join(CC, "windio_converter", "validation")

# --- regime thresholds (percent) ---
# Calibrated to OUR measured IEA-22 data: t_max/h 7.2% (r=0.9) -> +7% err, 9.2% (r=0.95) -> +24% err.
# The knee sits between 7 and 9%, so SOLID-CHECK fires at 8%.
TH_FULL   = 5.0    # t/h below this: RM trusted on ALL 6x6 terms (measured <=~5%)
TH_SOLID  = 8.0    # t/h at/above this: transverse shear (GA2/GA3) unreliable -> keep solid (mh104 +133%, IEA tip +24%)
TCORE_OK  = 5.0    # soft-core / section-height below this: sandwich is a minor (secondary) load path
TCORE_SOLID = 8.0  # soft-core / section-height at/above this: sandwich shear matters at section level
G_SOFT    = 1.0e9  # material Gmin below this = soft core (foam ~5e7; all structural plies >= 1.3e9)
CORE_FRAC = 0.30   # a wall is a "sandwich" if a soft ply is >= this fraction of the wall thickness


def parse_stations(case):
    """Return {tag: {'r':, 'chord':, 'airf_h':, 't_max_h':}} from the GEOMETRY block only of
    <case>_stations.dat (the file also holds RM/KL/solid diagonal + %err blocks with the same r leading
    column -- must not read those)."""
    path = os.path.join(VAL, "%s_stations.dat" % case)
    out = {}
    in_geom = False
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if "geometry & thin-wall" in s:
                in_geom = True; continue
            if in_geom and s.startswith("#") and "---" in s:
                break                                  # reached the next block -> stop
            if not in_geom or not s or s.startswith("#") or s[0].isalpha():
                continue
            p = s.split()
            if len(p) < 9:
                continue
            r = float(p[0]); tag = "r%03d" % round(r * 100)
            out[tag] = {"r": r, "chord": float(p[1]), "airf_h": float(p[4]), "t_max_h": float(p[8])}
    return out


def soft_core(shell_yaml):
    """Thickest soft-core sandwich in any wall. Returns (has_sandwich, core_mm, core_frac, worst_set)."""
    d = yaml.safe_load(open(shell_yaml))
    gmin = {m["name"]: min(m["elastic"]["G"]) for m in d["materials"]}
    secs = d.get("sections") or d.get("Sections") or []
    best = (False, 0.0, 0.0, "")
    for s in secs:
        layup = s.get("layup") or s.get("layers") or []
        tot = sum(float(p[1]) for p in layup)
        if tot <= 0:
            continue
        core = sum(float(p[1]) for p in layup if gmin.get(p[0], 1e12) < G_SOFT)
        if core > 0 and core / tot >= CORE_FRAC and core > best[1]:
            best = (True, core, core / tot, s.get("elementSet", "?"))
    return best


def parse_measured(case):
    """Measured RM max|diagonal %err| vs 2D-solid per tag, from the '%err vs 2D-solid (RM / KL)' block of
    <case>_stations.dat (cols RM_EA..RM_EI3). Returns {} if the blade has no solid reference yet."""
    path = os.path.join(VAL, "%s_stations.dat" % case)
    out, in_blk = {}, False
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if "%err vs 2D-solid" in s:
                in_blk = True; continue
            if in_blk and s.startswith("#") and "---" in s:
                break
            if not in_blk or not s or s.startswith("#") or s[0].isalpha():
                continue
            p = s.split()
            if len(p) < 7:
                continue
            tag = "r%03d" % round(float(p[0]) * 100)
            out[tag] = max(abs(float(x)) for x in p[1:7])      # RM diagonal terms
    return out


def verdict(t_max_h, t_core_h):
    """Map (t_max/h %, t_core/h %) -> (label, note). Sandwich PRESENCE alone does not escalate -- only a
    soft core that is a large fraction of section height (t_core/h) does, per the section-level shear theory."""
    if t_max_h >= TH_SOLID or t_core_h >= TCORE_SOLID:
        why = []
        if t_max_h >= TH_SOLID:    why.append("thick wall t/h>=%.0f%%" % TH_SOLID)
        if t_core_h >= TCORE_SOLID: why.append("soft core t_core/h>=%.0f%%" % TCORE_SOLID)
        return "SOLID-CHECK", "keep 2D-solid for GA2/GA3 (+ junction terms); " + ", ".join(why)
    if t_max_h >= TH_FULL or t_core_h >= TCORE_OK:
        why = []
        if t_max_h >= TH_FULL:   why.append("t/h %.1f%%" % t_max_h)
        if t_core_h >= TCORE_OK: why.append("t_core/h %.1f%%" % t_core_h)
        return "RM-OK*shear-watch", "EA/EI/GJ trusted; GA2/GA3 may be ~5-10% off (" + ", ".join(why) + ")"
    return "RM-FULL", "trust all 6x6 terms"


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "iea22"
    stn = parse_stations(case)
    meas = parse_measured(case)            # measured RM max-diag %err, if a solid reference exists
    rows = []
    for tag in sorted(stn):
        st = stn[tag]
        sh = os.path.join(VAL, "shell_%s_%s.yaml" % (case, tag))
        has, core_m, frac, wset = soft_core(sh) if os.path.exists(sh) else (False, 0.0, 0.0, "")
        t_core_h = 100.0 * (core_m / st["airf_h"]) if st["airf_h"] > 0 else 0.0
        lab, note = verdict(st["t_max_h"], t_core_h)
        rows.append((st["r"], st["t_max_h"], t_core_h, core_m * 1000, has, lab, note, meas.get(tag)))

    hdr = "  r    t_max/h[%]  t_core/h[%]  core[mm]  sandwich   verdict            meas.RM%err  note"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    out = ["# %s RM-regime guard: can RM 1D-shell REPLACE the 2D-solid at this station?" % case,
           "# thresholds: t/h<%.0f%%=RM-FULL, %.0f-%.0f%%=shear-watch, >=%.0f%%=SOLID-CHECK; soft-core (Gmin<%.1eGPa)"
           % (TH_FULL, TH_FULL, TH_SOLID, TH_SOLID, G_SOFT / 1e9),
           "# t_core/h = soft-core thickness / airfoil height (sandwich shear relevance at section level)",
           "# r   t_max/h%  t_core/h%  core_mm  sandwich  verdict  note"]
    for (r, tmh, tch, cmm, has, lab, note, me) in rows:
        mestr = "%6.1f" % me if me is not None else "   n/a"
        print("  %.2f   %7.2f     %7.2f     %6.1f   %-5s     %-18s %s     %s" % (
            r, tmh, tch, cmm, "YES" if has else "no", lab, mestr, note))
        out.append("%.2f  %.2f  %.2f  %.1f  %d  %s  %s  %s" % (
            r, tmh, tch, cmm, int(has), lab, mestr.strip(), note))

    n_full = sum(1 for x in rows if x[5] == "RM-FULL")
    n_watch = sum(1 for x in rows if x[5].startswith("RM-OK"))
    n_solid = sum(1 for x in rows if x[5] == "SOLID-CHECK")
    summary = "\nSUMMARY: %d RM-FULL, %d shear-watch, %d SOLID-CHECK  (of %d stations)" % (
        n_full, n_watch, n_solid, len(rows))
    print(summary)
    out.append("#" + summary.strip())
    with open(os.path.join(VAL, "%s_rm_regime.dat" % case), "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote %s_rm_regime.dat -> validation/" % case)


if __name__ == "__main__":
    main()
