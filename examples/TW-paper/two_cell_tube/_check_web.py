import yaml
import numpy as np
D = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code\multicell_tube\data"


def rows(r, n=2):
    return [float(v) for v in (str(r[0]).split() if len(r) == 1 else r)][:n]


sd = yaml.safe_load(open(D + r"\solid_tube2cell_thick.yaml"))
sn = np.array([rows(r) for r in sd["nodes"]])
# thin equatorial slice |y|<0.003 ; web is the cluster near x=0 (walls are at |x|~0.05)
sl = sn[np.abs(sn[:, 1]) < 0.003]
webx = np.sort(sl[np.abs(sl[:, 0]) < 0.02, 0])
print("t = 0.016, R = 0.05")
print("SOLID web nodes at equator, x sorted:", np.round(webx, 4))
print("  web x-extent: [%.4f, %.4f]  width=%.4f  center=%.4f"
      % (webx.min(), webx.max(), webx.max() - webx.min(), 0.5 * (webx.min() + webx.max())))

hd = yaml.safe_load(open(D + r"\tube2cell_thick.yaml"))
hn = np.array([rows(r) for r in hd["nodes"]])
hsl = hn[(np.abs(hn[:, 1]) < 0.01) & (np.abs(hn[:, 0]) < 0.02)]
print("SHELL web line x near equator:", np.unique(np.round(hsl[:, 0], 4)))
