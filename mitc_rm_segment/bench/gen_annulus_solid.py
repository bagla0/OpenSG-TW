"""Generate a prismatic thick-tube (annulus) 2D-solid cross-section in the
OpenSG-1.0 Solid_2DSG YAML format, matching the shell iso tube (R=1, t=0.1,
E=70 GPa, nu=0.3).  Beam axis = e1 = (0,0,1); cross-section in (x,y)."""
import numpy as np, yaml, math

R, t = 1.0, 0.1
ri, ro = R - t / 2, R + t / 2
NC, NR = 160, 4                      # hoop divisions, radial layers
E, nu = 70e9, 0.3
G = E / (2 * (1 + nu))

rs = np.linspace(ri, ro, NR + 1)
th = np.linspace(0.0, 2 * np.pi, NC, endpoint=False)
nodes = []
for ir in range(NR + 1):
    for ic in range(NC):
        x, y = rs[ir] * math.cos(th[ic]), rs[ir] * math.sin(th[ic])
        nodes.append("%.9f %.9f %.9f" % (x, y, 0.0))

def nid(ir, ic):                     # 1-based node id
    return ir * NC + (ic % NC) + 1

elements, orients = [], []
for ir in range(NR):
    for ic in range(NC):
        elements.append("%d %d %d %d" % (nid(ir, ic), nid(ir, ic + 1),
                                         nid(ir + 1, ic + 1), nid(ir + 1, ic)))
        tc = th[ic] + 0.5 * (th[1] - th[0])            # element-center hoop angle
        e1 = [0.0, 0.0, 1.0]
        e2 = [math.cos(tc), math.sin(tc), 0.0]         # radial
        e3 = [-math.sin(tc), math.cos(tc), 0.0]        # hoop tangent
        orients.append(e1 + e2 + e3)

sets = {"element": [{"name": "iso", "labels": list(range(1, len(elements) + 1))}]}
materials = [{"name": "iso", "E": [E, E, E], "G": [G, G, G],
             "nu": [nu, nu, nu], "rho": 2700.0}]
doc = {"nodes": [[s] for s in nodes], "elements": [[s] for s in elements],
       "sets": sets, "elementOrientations": orients, "materials": materials}
out = r"C:\Users\bagla0\AppData\Local\Temp\claude\C--Users-bagla0\91cf4f05-ed42-47e2-974c-813d98a91247\scratchpad\annulus_tube.yaml"
with open(out, "w") as f:
    yaml.safe_dump(doc, f, default_flow_style=None, sort_keys=False)
A = math.pi * (ro**2 - ri**2); J = math.pi * (ro**4 - ri**4) / 2; I = math.pi * (ro**4 - ri**4) / 4
print("wrote", out, " nodes", len(nodes), " elems", len(elements))
print("analytic solid annulus: EA=%.4e GJ=%.4e EI=%.4e" % (E * A, G * J, E * I))
