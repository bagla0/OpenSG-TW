"""TW REGRESSION GUARDRAIL -- must pass for EVERY transverse-shear / RM solver change.

Single-cell AND multi-cell thin-walled benchmarks where JAX-RM is validated against the
FEniCS 2D-solid. Compares the legacy 'reduced' integration (which produced the published
numbers) against 'mitc' and the production-default 'mitc_both' (tie-both) assumed-strain
schemes. A change PASSES iff, on every TW case, mitc_both is no LESS accurate than the validated
'reduced' result against the solid (by more than TOL) on any diagonal term AND RM is still at
least as good as KL on the transverse-shear terms (GA2,GA3). (mitc_both is EXPECTED to improve
GA3 on thick walls, so the criterion bounds accuracy REGRESSION, not drift from reduced.)

Also reports the composite (soft-core) TARGET: MITC should improve GA2 there.

Run:  python tw_regression_guardrail.py
"""
import os
import sys
import numpy as np
import yaml

CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
sys.path.insert(0, CC)
import jax
jax.config.update("jax_enable_x64", True)
from fe_jax.strip_RM import rm_timoshenko_6x6
from fe_jax.gradient_kirchhoff import gradient_junction_kirchhoff

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
TOL = 1.5   # % RM(mitc_both) must stay within this of RM(reduced) on every diagonal term per TW case

# (label, folder, mesh, solid, curved, dshift_mode)  dshift: 'half'=T/2 from layup, 'none'=None.
# Case dirs live under tests/research/<folder>/data/.
CASES = [
    ("SINGLE box 1-cell", "multicell_box", "box_shell_1cell.yaml", "C6_solid_box1.txt", False, "half"),
    ("SINGLE tube [-45]", "tube_thesis_314", "shell_center.yaml", "C6_solid_314.txt", True, "none"),
    ("MULTI  box 2-cell", "multicell_box", "box_shell_2cell.yaml", "C6_solid_box2.txt", False, "half"),
    ("MULTI  box 2-cell thick", "multicell_box", "box_shell_2cell_thick.yaml", "C6_solid_box2_thick.txt", False, "half"),
    ("MULTI  tube 2-cell thin", "multicell_tube", "tube2cell_thin.yaml", "C6_solid_tube2cell_thin.txt", True, "half"),
    ("MULTI  tube 2-cell thick", "multicell_tube", "tube2cell_thick.yaml", "C6_solid_tube2cell_thick.txt", True, "half"),
]


def sym(M):
    M = np.asarray(M)
    return 0.5 * (M + M.T)


def wall_t(meshp):
    d = yaml.safe_load(open(meshp))
    return sum(float(p[1]) for p in d["sections"][0]["layup"])


def pe(m, s):
    return 100.0 * (m - s) / s if s != 0 else float("nan")


def cdir(folder):
    return os.path.join(CC, "tests", "research", folder, "data")


print("=" * 104)
print("TW REGRESSION GUARDRAIL  (RM 'reduced' [validated] vs 'mitc' vs 'mitc_both' [default])   TOL=%.1f%%" % TOL)
print("=" * 104)
all_pass = True
for label, folder, mesh, solidf, curved, dsm in CASES:
    mp = os.path.join(cdir(folder), mesh)
    sp = os.path.join(cdir(folder), solidf)
    if not (os.path.exists(mp) and os.path.exists(sp)):
        print("\n%-26s  SKIP (missing mesh/solid)" % label)
        continue
    S = sym(np.loadtxt(sp))
    T = wall_t(mp)
    ds = T / 2.0 if dsm == "half" else None
    KL = sym(gradient_junction_kirchhoff(mp, frac=0.0, dshift=ds, orient=False)[0])
    Rr = sym(rm_timoshenko_6x6(mp, 0.0, dshift=ds, curved=curved, shear="reduced", v1shear="reduced", orient=False))
    Rm = sym(rm_timoshenko_6x6(mp, 0.0, dshift=ds, curved=curved, shear="mitc", v1shear="mitc", orient=False))
    Rb = sym(rm_timoshenko_6x6(mp, 0.0, dshift=ds, curved=curved, shear="mitc_both", v1shear="mitc_both", orient=False))
    print("\n%-26s  (t=%.4g, dshift=%s, curved=%s)" % (label, T, "%.4g" % ds if ds else "None", curved))
    print("   term     solid        KL %     RM_reduced %    RM_mitc %   RM_mitc_both %   d(both-red)")
    case_pass = True
    for i in range(6):
        kl, rr, rm, rb = pe(KL[i, i], S[i, i]), pe(Rr[i, i], S[i, i]), pe(Rm[i, i], S[i, i]), pe(Rb[i, i], S[i, i])
        d = rb - rr                           # signed change of production default vs validated reduced
        regress = abs(rb) - abs(rr)           # >0 => mitc_both is LESS accurate than reduced (vs solid)
        flag = "" if regress <= TOL else "  <-- REGRESS"
        if regress > TOL:
            case_pass = False
        print("   %-5s %11.4e  %+7.2f   %+9.2f     %+8.2f    %+9.2f     %+7.2f%s"
              % (LBL[i], S[i, i], kl, rr, rm, rb, d, flag))
    # production-default RM (mitc_both) must still be at least as good as KL on the shear terms (GA2,GA3)
    shear_ok = all(abs(pe(Rb[i, i], S[i, i])) <= abs(pe(KL[i, i], S[i, i])) + TOL for i in (1, 2))
    verdict = "PASS" if (case_pass and shear_ok) else "FAIL"
    if verdict == "FAIL":
        all_pass = False
    print("   -> %s  (mitc_both no accuracy regression >%.1f%% vs reduced: %s ; RM<=KL on shear: %s)"
          % (verdict, TOL, case_pass, shear_ok))

print("\n" + "=" * 104)
print("GUARDRAIL RESULT:  %s" % ("ALL PASS" if all_pass else "FAIL -- do not ship this change"))
print("=" * 104)

# ---- composite (soft-core) TARGET: MITC should improve GA2 ----
cm = os.path.join(cdir("mh104_center_ref"), "shell_mh104_mid_f020.yaml")
cs = os.path.join(cdir("mh104_center_ref"), "C6_solid_mh104_center_f020.txt")
if os.path.exists(cm) and os.path.exists(cs):
    Sc = sym(np.loadtxt(cs))
    rr = sym(rm_timoshenko_6x6(cm, 0.5, shear="reduced", v1shear="reduced", orient=False))
    rm = sym(rm_timoshenko_6x6(cm, 0.5, shear="mitc", v1shear="mitc", orient=False))
    rb = sym(rm_timoshenko_6x6(cm, 0.5, shear="mitc_both", v1shear="mitc_both", orient=False))
    print("\nCOMPOSITE soft-core TARGET (mh104 thin, frac=0.5):")
    for i in (1, 2):
        print("   %-4s  RM_reduced %+7.2f%%   RM_mitc %+7.2f%%   RM_mitc_both %+7.2f%%   (solid %.4e)"
              % (LBL[i], pe(rr[i, i], Sc[i, i]), pe(rm[i, i], Sc[i, i]), pe(rb[i, i], Sc[i, i]), Sc[i, i]))
