"""Reduce the FEA and RM membrane-N fields to per-station section moments and compare to the applied
beam force FF.  The N that integrates to FF carries the load correctly (e1=span, so span-N = Nvec[:,0])."""
import os, numpy as np
D = os.path.join(os.path.dirname(__file__), "..", "data")
m = np.load(os.path.join(D, "blade_mesh.npz")); nodes = m["nodes"]; quads = m["quads"]
fea = np.load(os.path.join(D, "blade_fea.npz")); rm = np.load(os.path.join(D, "blade_rm.npz"))
Nf = fea["Nvec"][:, 0]; Nr = rm["Nvec"][:, 0]; FF = rm["FF"]                       # span-N (e1=span)
MPER = 2; NS = 101; NSE = len(quads) // (NS - 1); BLADE_LEN = 138.204


def section_MF(Nsp, p):
    els = np.arange(p * NSE, (p + 1) * NSE); Fx = 0.0; My = 0.0
    for e in els:
        q = quads[e]; X = nodes[q]
        Lsec = np.linalg.norm(X[3] - X[0])                                          # arc edge (0->3) length
        zc = X[:, 2].mean()
        Fx += Nsp[e] * Lsec; My += Nsp[e] * Lsec * zc
    return Fx, My


print("stn   X     FF.My(applied)   FEA int    RM int   |  FEA/FF   RM/FF")
for i in [0, 5, 10, 25, 40]:
    p = min(i * MPER, NS - 2)
    Ff, Mf = section_MF(Nf, p); Fr, Mr = section_MF(Nr, p)
    print("  s%02d %5.1f   %+.3e   %+.3e %+.3e |  %.3f   %.3f"
          % (i, i / 50 * BLADE_LEN, FF[i, 4], Mf, Mr, Mf / (FF[i, 4] + 1e-30), Mr / (FF[i, 4] + 1e-30)))
