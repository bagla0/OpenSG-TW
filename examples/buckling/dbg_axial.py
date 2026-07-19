import os, numpy as np
OUT = "/home/roger/a/bagla0/OpenSG-TW-claude/examples/TW-paper/fsm_buckling"
d = np.load(os.path.join(OUT, "data", "cyl_iso_modes.npz"))
modes, nodes = d["fea_modes"], d["nodes"]
NC = 240; NL = nodes.shape[0] // NC - 1
rr = np.hypot(nodes[:, 1], nodes[:, 2]) + 1e-30
print("NL=%d NC=%d" % (NL, NC))
for c in range(2):
    ur = ((modes[:, 1, c] * nodes[:, 1] + modes[:, 2, c] * nodes[:, 2]) / rr).reshape(NL + 1, NC)
    ring = ur[NL // 2, :]
    cs = np.abs(np.fft.rfft(ring)); n_circ = int(np.argmax(cs[1:])) + 1
    pw = np.sum(np.abs(np.fft.rfft(ur, axis=0))**2, axis=1)
    top = (np.argsort(pw[1:])[::-1][:4] + 1)
    print("mode%d  n_circ=%d  axial top bins=%s" % (c + 1, n_circ, top))
    # count sign changes down the peak column
    col = ur[:, int(np.argmax(np.abs(ur).max(0)))]
    sc = int(np.sum(np.abs(np.diff(np.sign(col))) > 0))
    print("        peak-column axial sign-changes=%d (=> ~%d half-waves)" % (sc, sc))
