"""Convert a PreVABS-generated VABS .sg file (mesh connectivity + element angles)
to a 2D-solid YAML for OpenSG FEniCS (same format as 2Dsolid_*.yaml).
Logic from training_data/.../Solid_VABS_store2Dyaml.py (the YAML-writing part)."""
import os
import sys
import numpy as np
import yaml

SG = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\PreVABS\prevabs-v2.0.1-examples\prevabs-v2.0.1-examples\ex_airfoil_r\mh104.sg"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    os.path.join(os.path.dirname(__file__), "2Dsolid_VABS_mh_104.yaml")              # theta1 + theta3 (correct)
OUT_T1 = OUT.replace(".yaml", "_t1only.yaml")                                        # theta1 only (converter as-given)
NAMES = {1: "gelcoat", 2: "nexus", 3: "db_frp", 4: "ud_frp", 5: "balsa"}   # from mh104.sg.mat
SIGN3 = +1.0   # sign of the theta3 fiber rotation about the ply normal (validated vs mh104.sg.K off-diagonals)


class FlowList(list):
    pass


yaml.add_representer(FlowList, lambda d, data: d.represent_sequence(
    "tag:yaml.org,2002:seq", data, flow_style=True))

data = np.loadtxt(SG, delimiter=",", skiprows=0, dtype=str)   # one string per non-blank line
grp = int(data[0].split()[1])
nnode, nelem, nphases = [int(i) for i in data[3].split()]
print("grp(layups)=%d  nnode=%d  nelem=%d  nphases=%d" % (grp, nnode, nelem, nphases))

points = []                                                   # (axial=0, x, y)
for i in range(nnode):
    dat = data[4 + i].split()
    points.append((0.0, float(dat[1]), float(dat[2])))

elem = []                                                     # connectivity, gmsh node order, 0-based
for i in range(nelem):
    dat = data[4 + nnode + i].split()
    nodes = [int(x) for x in dat[1:] if int(x) != 0]          # drop VABS zero-padding (tri = 3, quad = 4)
    elem.append([n - 1 for n in nodes])
ncnt = sorted(set(len(e) for e in elem))
print("element node-counts present:", ncnt, "(3=triangle, 4=quad)")

p = 4 + nnode + 2 * nelem                                     # layup-group -> (material, theta3 fiber angle)
mat = [int(data[p + i].split()[1]) - 1 for i in range(grp)]
grp_theta3 = [float(data[p + i].split()[2]) for i in range(grp)]
print("\nlayup groups (group -> material, theta3 fiber angle):")
for g in range(grp):
    print("  group %d -> %-8s  theta3=%.0f deg" % (g + 1, NAMES.get(mat[g] + 1, "?"), grp_theta3[g]))

p = 4 + nnode + nelem                                         # per-element: id, layup-group, theta1 (contour angle)
sub, ang, ang3 = [], [], []
for row in range(nelem):
    c1, c2, c3 = data[p + row].split()
    g = int(c2)
    sub.append(mat[g - 1])
    ang.append(float(c3))                                     # theta1: ply-plane angle in cross-section (per element)
    ang3.append(grp_theta3[g - 1])                            # theta3: fiber angle within ply plane (per group)

p = 4 + nnode + 2 * nelem + grp                               # materials: 5 rows/phase (E, G, nu, density)
material_parameters, density = [], []
for ii in range(nphases):
    m = [data[p + 5 * ii + 1].split(), data[p + 5 * ii + 2].split(), data[p + 5 * ii + 3].split()]
    density.append(float(data[p + 5 * ii + 4].split()[0]))
    material_parameters.append(np.array(m, dtype=float).flatten().tolist())

print("\nparsed materials (phase: E, G, nu) -- check vs materials.xml:")
for i in range(nphases):
    mp = material_parameters[i]
    print("  phase %d (%-8s): E=[%.3e %.3e %.3e] G=[%.3e %.3e %.3e] nu=[%.2f %.2f %.2f]" % (
        i + 1, NAMES.get(i + 1, "?"), mp[0], mp[1], mp[2], mp[3], mp[4], mp[5], mp[6], mp[7], mp[8]))

def base_segment():
    seg = {"nodes": [], "elements": [], "sets": {"element": []}, "elementOrientations": [], "materials": []}
    for nd in points:
        seg["nodes"].append(FlowList(["%.6f %.6f %.6f" % (nd[1], nd[2], nd[0])]))
    for el in elem:
        seg["elements"].append(FlowList([" ".join(str(n + 1) for n in el)]))
    for mat_idx in sorted(set(sub)):
        labels = [int(i + 1) for i, s in enumerate(sub) if s == mat_idx]
        seg["sets"]["element"].append({"name": "Material_%d" % (mat_idx + 1), "labels": FlowList(labels)})
    for i in range(nphases):
        mp = material_parameters[i]
        seg["materials"].append({"name": "Material_%d" % (i + 1),
                                 "E": FlowList([mp[0], mp[1], mp[2]]),
                                 "G": FlowList([mp[3], mp[4], mp[5]]),
                                 "nu": FlowList([mp[6], mp[7], mp[8]]),
                                 "rho": float(density[i])})
    return seg


def frame(t1, t3):
    """Material frame [e1(fiber), e2, e3(ply normal)] as a 9-list.
    theta1 sets the ply-plane in the x-y cross-section (e3 = normal, e2 = tangent, e1 = beam axis Z);
    theta3 then rotates the fiber about the ply normal e3, tilting e1 from Z toward the tangent e2."""
    c1, s1 = np.cos(np.deg2rad(t1)), np.sin(np.deg2rad(t1))
    c3, s3 = np.cos(np.deg2rad(SIGN3 * t3)), np.sin(np.deg2rad(SIGN3 * t3))
    e1 = [s3 * c1, s3 * s1, c3]
    e2 = [c3 * c1, c3 * s1, -s3]
    e3 = [-s1, c1, 0.0]
    return FlowList([float(v) for v in e1 + e2 + e3])


seg = base_segment()
seg["elementOrientations"] = [frame(t1, t3) for t1, t3 in zip(ang, ang3)]
with open(OUT, "w") as f:
    yaml.dump(seg, f, sort_keys=False, default_flow_style=False)
print("\nwrote (theta1+theta3, correct):", OUT)

segt1 = base_segment()
segt1["elementOrientations"] = [frame(t1, 0.0) for t1 in ang]
with open(OUT_T1, "w") as f:
    yaml.dump(segt1, f, sort_keys=False, default_flow_style=False)
print("wrote (theta1 only, converter as-given):", OUT_T1)

nz = sum(1 for t in ang3 if abs(t) > 1e-6)
print("\nelements with non-zero theta3 (off-axis fiber): %d / %d (%.1f%%)" % (nz, nelem, 100.0 * nz / nelem))
