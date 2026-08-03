import numpy as np

for tag in ("oml", "mid"):
    z = np.load("data/final_%s_fields.npz" % tag)
    w = z["is_web"][z["el_gauss"]]
    sv = z["stress_vabs"]
    r = lambda x: float(np.sqrt(np.mean(x ** 2)))
    print(tag, "VABS signal rms: s11 web %.1f skin %.1f, s12 web %.1f skin %.1f"
          % (r(sv[w, 0]), r(sv[~w, 0]), r(sv[w, 5]), r(sv[~w, 5])))
