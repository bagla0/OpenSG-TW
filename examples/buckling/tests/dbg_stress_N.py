"""Does the step-1 mid-ref shell resultant N = A*s6[:3] + B*s6[3:6] equal the through-thickness integral
of the step-2 3D dehomogenized stress int(sigma_11) dz?  They must, by CLT -- unless the emitted mid-ref
ABD is inconsistent with the homogenization.  Composite IEA station, mixed axial + flap-moment load."""
import os, sys
import numpy as np
ROOT = "/home/roger/a/bagla0/OpenSG-TW-claude"
XSEC = os.path.join(ROOT, "examples", "TW-paper", "xsec_paper")
IEA = os.path.join(ROOT, "examples", "data", "iea_all_stations")
sys.path.insert(0, ROOT); sys.path.insert(0, XSEC)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax; jax.config.update("jax_enable_x64", True)
import dehom_rm
from emit_abd import load_station_abd

i = 10
shell = os.path.join(IEA, "shell51", "1d_yaml", "iea_s%02d_shell.yaml" % i)
B = dehom_rm.build_rm_bundle(shell)
abd = load_station_abd(os.path.join(IEA, "dehom51", "out", "abd", "iea_s%02d_abd.yaml" % i))["by_name"]
FF = [1.0e6, 0, 0, 0, 5.0e7, 0]                              # axial + flap moment (VABS order)
st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF)
corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"]); cen = corners.mean(0)
layups = B["layup_per_elem"]

print("elem  layup       N11_step1(A@s6+B@k)   N11_integral(int s11 dz)   ratio  int/step1")
for e in [0, 25, 55, 90, 130]:
    if e >= len(rc):
        continue
    c0, c1 = int(rc[e, 0]), int(rc[e, 1]); mid = 0.5 * (corners[c0] + corners[c1])
    s6, _ = dehom_rm._rm_shell_strain(B, e, 0.5, st_m, aA, aB)
    A6, G, h = abd[layups[e]]; A6 = np.asarray(A6)
    N_step1 = A6[:3, :3] @ s6[:3] + A6[:3, 3:] @ s6[3:6]
    t2, t3 = corners[c1] - corners[c0]; tl = np.hypot(t2, t3); t2, t3 = t2 / tl, t3 / tl
    n2, n3 = t3, -t2
    if (cen[0] - mid[0]) * n2 + (cen[1] - mid[1]) * n3 < 0:
        n2, n3 = -n2, -n3                                    # inward normal
    zc = np.linspace(-h / 2, h / 2, 41)                      # mid-ref depth across the wall
    pts = np.array([[mid[0] + z * n2, mid[1] + z * n3] for z in zc])
    S = np.asarray(dehom_rm.stress_at_points(B, pts, beam_force_vabs=FF, frame="global", n_per_layer=4)["stress"])
    N11_int = np.trapz(S[:, 0], zc)                          # int sigma_11 dz
    print("  %-4d  %-10s  %+.5e        %+.5e       %.3f" % (e, layups[e], N_step1[0], N11_int, N11_int / N_step1[0]))
