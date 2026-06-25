"""Decide the thin/thick wall boundary factor f for mh104 from geometry: the classical thin-shell
limit is wall thickness h <~ 0.1 x local section dimension H.  Use the SPAR cap (thickest,
stiffness-dominant laminate) and the local airfoil thickness there."""
import os
import numpy as np

DP = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\training data\opensg-FEniCS\data\mh104_training\prevabs to ymal\datapoints_mh104.txt"
TH = dict(gelcoat=0.000381, nexus=0.00051, db_frp=0.00053, ud_frp=0.00053, balsa=0.003125)
# full (f=1) laminate thicknesses
spar = [("gelcoat", 1), ("nexus", 1), ("db_frp", 17), ("ud_frp", 38), ("balsa", 1), ("ud_frp", 37), ("db_frp", 16)]
LE = [("gelcoat", 1), ("nexus", 1), ("db_frp", 18)]
h_spar = sum(TH[m] * n for m, n in spar)
h_le = sum(TH[m] * n for m, n in LE)

pts = np.array([[1.9 * (float(p.split()[1]) - 0.25), 1.9 * float(p.split()[2])]
                for p in open(DP).read().splitlines() if p.strip() and p.split()[0] not in
                ("l12", "l23", "l34", "h34", "h23", "h12")])
x_spar = 1.9 * (0.511 - 0.25)            # rear spar web, model x
near = pts[np.abs(pts[:, 0] - x_spar) < 0.06]
H_spar = near[:, 1].max() - near[:, 1].min()
x_mid = 1.9 * (0.30 - 0.25)              # ~max-thickness location
near2 = pts[np.abs(pts[:, 0] - x_mid) < 0.06]
H_max = near2[:, 1].max() - near2[:, 1].min()

print("full-thickness (f=1) laminate:  spar cap h=%.4f m   LE h=%.4f m" % (h_spar, h_le))
print("local airfoil thickness:  at spar H=%.4f m   at max-thickness H=%.4f m" % (H_spar, H_max))
print("\nh/H at the spar cap vs f (thin-shell limit ~0.1):")
for f in (0.1, 0.2, 0.3, 0.4, 0.6, 1.0):
    print("  f=%.1f   h/H_spar=%.3f   h/H_max=%.3f" % (f, h_spar * f / H_spar, h_spar * f / H_max))
print("\nf at which spar h/H = 0.10 :  H_spar -> f=%.2f ,  H_max -> f=%.2f" %
      (0.10 * H_spar / h_spar, 0.10 * H_max / h_spar))
