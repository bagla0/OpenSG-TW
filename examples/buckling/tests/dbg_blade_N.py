"""Confirm the FEA vs RM N frame mismatch: FEA local frame e1=arc,e2=span; RM ring frame 1=span,2=arc."""
import os, numpy as np
D = os.path.join(os.path.dirname(__file__), "..", "data")
fea = np.load(os.path.join(D, "blade_fea.npz")); rm = np.load(os.path.join(D, "blade_rm.npz"))
nf, nr = fea["Nvec"], rm["Nvec"]
print("FEA  N[:,0] (arc)   range [%.3e, %.3e]  rms %.3e" % (nf[:, 0].min(), nf[:, 0].max(), np.sqrt((nf[:, 0]**2).mean())))
print("FEA  N[:,1] (span)  range [%.3e, %.3e]  rms %.3e" % (nf[:, 1].min(), nf[:, 1].max(), np.sqrt((nf[:, 1]**2).mean())))
print("RM   N[:,0] (span)  range [%.3e, %.3e]  rms %.3e" % (nr[:, 0].min(), nr[:, 0].max(), np.sqrt((nr[:, 0]**2).mean())))
print("RM   N[:,1] (arc)   range [%.3e, %.3e]  rms %.3e" % (nr[:, 1].min(), nr[:, 1].max(), np.sqrt((nr[:, 1]**2).mean())))
# correct comparison: FEA span (col1) vs RM span (col0)
a, b = nf[:, 1], nr[:, 0]
print("\nCORRECT match FEA_span vs RM_span : ||RM-FEA||/||FEA|| = %.3f  corr=%.3f"
      % (np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-30), np.corrcoef(a, b)[0, 1]))
