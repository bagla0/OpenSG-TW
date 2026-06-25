"""Build a .dat record over ALL IEA-22 stations: geometry + thin-wall metric + Timoshenko 6x6 diagonal
(RM / KL / 2D-solid) and %err. Also answers 'is it thin-walled?' via t_wall / airfoil-height."""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
import windIO
from windio_to_opensg import WindIOBlade, build_cross_section
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff

VAL = os.path.join(CC, "windio_converter", "validation")
blade = WindIOBlade(os.path.join(os.path.dirname(windIO.__file__), "examples", "turbine", "IEA-22-280-RWT.yaml"))
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
STATIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


rows = []
for r in STATIONS:
    tag = "r%03d" % round(r * 100)
    cs = build_cross_section(blade, r, mesh_size=0.01)
    chord, twist = cs["chord"], cs["twist"]
    rthick = blade.scalar("rthick", r)
    h = rthick * chord                                       # airfoil thickness (section height)
    inv = {v: k for k, v in cs["laminates"].items()}
    lam_t = [sum(t for (m, t, a) in inv[k]) for k in range(len(inv))]
    t_max = max(lam_t); t_min = min(lam_t)                   # thickest (spar cap) / thinnest (LE/TE skin) wall
    shell = os.path.join(VAL, "shell_iea22_%s.yaml" % tag)
    RM = sym(rm_timoshenko_6x6(shell, 0.0, orient=False))
    KL = sym(gradient_junction_kirchhoff(shell, frac=0.0, orient=False)[0])
    sp = os.path.join(VAL, "C6_solid_iea22_%s.txt" % tag)
    S = sym(np.loadtxt(sp)) if os.path.exists(sp) else None
    rows.append(dict(r=r, tag=tag, chord=chord, twist=twist, rthick=rthick, h=h,
                     t_max=t_max, t_min=t_min, RM=RM, KL=KL, S=S))

# ---- write the .dat record ----
out = os.path.join(VAL, "iea22_stations.dat")
with open(out, "w") as f:
    f.write("# IEA-22-280-RWT windIO cross-section homogenization record  (OML reference, frac=0)\n")
    f.write("# geometry + THIN-WALL metric (t_max = thickest wall = spar cap; t_min = thinnest = LE/TE skin):\n")
    f.write("#  r  chord[m]  twist[deg]  rthick  airfoil_h[m]  t_min[mm]  t_max[mm]  t_max/chord[%]  t_max/h[%]\n")
    f.write("# Timoshenko 6x6 diagonal [EA GA2 GA3 GJ EI2 EI3] for RM, KL, 2D-solid; and max|diag %err| vs solid\n#\n")
    f.write("# --- geometry & thin-wall ---\n")
    f.write("%-6s %9s %10s %8s %11s %9s %9s %13s %11s\n"
            % ("r", "chord", "twist", "rthick", "airf_h", "t_min_mm", "t_max_mm", "t_max/c[%]", "t_max/h[%]"))
    for d in rows:
        f.write("%-6.2f %9.3f %10.2f %8.3f %11.3f %9.1f %9.1f %13.2f %11.2f\n"
                % (d["r"], d["chord"], d["twist"], d["rthick"], d["h"], d["t_min"] * 1e3, d["t_max"] * 1e3,
                   100 * d["t_max"] / d["chord"], 100 * d["t_max"] / d["h"]))
    for model in ("RM", "KL", "S"):
        nm = {"RM": "JAX MSG-RM", "KL": "JAX MSG-Kirchhoff", "S": "FEniCS-2D-solid"}[model]
        f.write("#\n# --- %s Timoshenko 6x6 diagonal ---\n" % nm)
        f.write("%-6s %12s %12s %12s %12s %12s %12s\n" % ("r", *LBL))
        for d in rows:
            M = d[model]
            if M is None:
                f.write("%-6.2f %12s\n" % (d["r"], "(solid pending)")); continue
            f.write("%-6.2f " % d["r"] + " ".join("%12.4e" % M[i, i] for i in range(6)) + "\n")
    f.write("#\n# --- diagonal %err vs 2D-solid (RM / KL) ---\n")
    f.write("%-6s %10s %10s %10s %10s %10s %10s   %10s %10s %10s %10s %10s %10s\n"
            % ("r", *["RM_" + L for L in LBL], *["KL_" + L for L in LBL]))
    for d in rows:
        if d["S"] is None:
            f.write("%-6.2f  (solid pending)\n" % d["r"]); continue
        er = [100 * (d["RM"][i, i] - d["S"][i, i]) / d["S"][i, i] for i in range(6)]
        ek = [100 * (d["KL"][i, i] - d["S"][i, i]) / d["S"][i, i] for i in range(6)]
        f.write("%-6.2f " % d["r"] + " ".join("%+10.2f" % e for e in er) + "   " + " ".join("%+10.2f" % e for e in ek) + "\n")

# ---- console summary ----
print("=== IEA-22 thin-wall metric (t_max = spar cap wall) ===")
print("  r    chord   airf_h   t_max[mm]  t_max/chord  t_max/h   thin-walled?")
tw_max = 0
for d in rows:
    th = 100 * d["t_max"] / d["h"]; tc = 100 * d["t_max"] / d["chord"]; tw_max = max(tw_max, th)
    print("  %.2f  %5.2f   %5.2f    %6.1f     %6.2f%%     %6.2f%%   %s"
          % (d["r"], d["chord"], d["h"], d["t_max"] * 1e3, tc, th, "YES" if th < 10 else "borderline"))
print("\n  max(t_max/h) over span = %.1f%%  ->  %s" % (tw_max, "THIN-WALLED (t/h << 1)" if tw_max < 12 else "thick at root"))
print("\n=== diagonal %err vs solid (where solid ready) ===")
print("  r     EA    GA2    GA3    GJ    EI2    EI3   (RM)  |  max|err| RM / KL")
for d in rows:
    if d["S"] is None:
        print("  %.2f   (solid pending)" % d["r"]); continue
    er = [100 * (d["RM"][i, i] - d["S"][i, i]) / d["S"][i, i] for i in range(6)]
    ek = [100 * (d["KL"][i, i] - d["S"][i, i]) / d["S"][i, i] for i in range(6)]
    print("  %.2f " % d["r"] + " ".join("%+6.1f" % e for e in er)
          + "   | RM %.1f  KL %.1f" % (max(abs(x) for x in er), max(abs(x) for x in ek)))
print("\nwrote", out)
