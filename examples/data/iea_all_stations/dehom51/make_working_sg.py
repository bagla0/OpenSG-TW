'''make_working_sg.py -- produce WORKING VABS .sg for the 3 problem stations:
  s00 : existing PreVABS tri mesh has ONE inverted triangle -> flip its winding (surgical, keeps mesh).
  s02, s50 : replace with the clean pyNuMAD-quad mesh (fallback_yaml, LE-based frame matching the 49),
             converted YAML -> VABS .sg using the exact .sg format + orientation convention.

VABS .sg layout (decoded from the good stations):
  1  nlayer  /  1 0 0  /  0 0 0 0  /  nnode nelem nmate
  [nnode]  nid x y
  [nelem]  eid n1..n9         (tri=3+6 zeros, quad=4+5 zeros; 1-based)
  [nelem]  eid layer_id theta1(deg)     # in-plane contour angle
  [nlayer] layer_id mat_id theta3(deg)  # ply angle (group)
  [nmate]  mat_id 1 / E1 E2 E3 / G12 G13 G23 / nu12 nu13 nu23 / rho   (orthotropic block; iso replicated)

Orientation -> angles is the exact inverse of convert_sg_to_yaml.frame():
  e3=[-s1,c1,0] -> theta1=atan2(-e3x,e3y);  c3=e1z, s3=e1x*c1+e1y*s1 -> theta3=atan2(s3,c3).
'''
import os
import numpy as np
import yaml

ROOT = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'
SGV = os.path.join(ROOT, 'shell51', 'sg_v201')
FB = os.path.join(ROOT, 'shell51', 'fallback_yaml')


def _row(x):
    if isinstance(x, list):
        if len(x) and isinstance(x[0], str):
            return [float(v) for v in x[0].split()]    # ["x y z"] flow-string form
        return [float(v) for v in x]                   # [x, y, z] list-of-floats form
    return [float(v) for v in str(x).split()]


# ---------------------------------------------------------------- .sg writer
def write_sg(path, xy, conn, elem_group, elem_theta1, group_mat_theta3, materials):
    nlayer = len(group_mat_theta3)
    nmate = len(materials)
    with open(path, 'w') as f:
        f.write('%8d%8d\n\n' % (1, nlayer))
        f.write('%8d%8d%8d\n\n' % (1, 0, 0))
        f.write('%8d%8d%8d%8d\n\n' % (0, 0, 0, 0))
        f.write('%8d%8d%8d\n\n' % (len(xy), len(conn), nmate))
        for i, (x, y) in enumerate(xy):
            f.write('%8d%20.9e%20.9e\n' % (i + 1, x, y))
        f.write('\n')
        for i, c in enumerate(conn):
            n9 = (list(c) + [0] * 9)[:9]
            f.write('%8d' % (i + 1) + ''.join('%8d' % n for n in n9) + '\n')
        f.write('\n')
        for i in range(len(conn)):
            f.write('%8d%8d%20.9e\n' % (i + 1, elem_group[i], elem_theta1[i]))
        f.write('\n')
        for g, (mid, th3) in enumerate(group_mat_theta3):
            f.write('%8d%8d%20.9e\n' % (g + 1, mid, th3))
        f.write('\n')
        for m, mat in enumerate(materials):
            f.write('%8d%8d\n' % (m + 1, 1))                       # orthotropic flag
            f.write('%20.9e%20.9e%20.9e\n' % tuple(mat['E']))
            f.write('%20.9e%20.9e%20.9e\n' % tuple(mat['G']))
            f.write('%20.9e%20.9e%20.9e\n' % tuple(mat['nu']))
            f.write('%20.9e\n' % mat['rho'])
        f.write('\n')
    # sibling .mat name map
    with open(path + '.mat', 'w') as f:
        for m, mat in enumerate(materials):
            f.write('%4d    %s\n' % (m + 1, mat['name']))
        f.write('\n')


def signed_area(p):
    s = 0.0
    for i in range(len(p)):
        x1, y1 = p[i]; x2, y2 = p[(i + 1) % len(p)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


# ---------------------------------------------------------------- s00: flip inverted triangles
def parse_sg(path):
    L = [l for l in open(path).read().splitlines() if l.strip()]
    h = next(i for i, l in enumerate(L)
             if len(l.split()) == 3 and all(x.lstrip('-').isdigit() for x in l.split()) and int(l.split()[0]) > 1000)
    nn, ne = int(L[h].split()[0]), int(L[h].split()[1])
    xy = np.array([[float(L[h + 1 + k].split()[1]), float(L[h + 1 + k].split()[2])] for k in range(nn)])
    conn = [[int(x) for x in L[h + 1 + nn + k].split()[1:] if int(x) != 0] for k in range(ne)]
    return L, h, nn, ne, xy, conn


def flip_sg00():
    src = os.path.join(SGV, 'iea_s00.sg')
    bak = src + '.orig'
    if not os.path.exists(bak):
        import shutil; shutil.copy(src, bak)
    L, h, nn, ne, xy, conn = parse_sg(bak)
    lines = open(bak).read().splitlines()
    # map: which raw line index is each element's connectivity? re-scan non-blank with same rule
    NB = [i for i, l in enumerate(lines) if l.strip()]
    conn_start_nb = h + 1 + nn                                    # index into non-blank list
    nflip = 0
    for e in range(ne):
        p = xy[[n - 1 for n in conn[e]]]
        if signed_area(p) < 0:
            raw = NB[conn_start_nb + e]
            t = lines[raw].split()
            eid = t[0]; nodes = [x for x in t[1:] if x != '0']
            nodes = [nodes[0]] + nodes[1:][::-1]                  # reverse winding, keep first vertex
            n9 = (nodes + ['0'] * 9)[:9]
            lines[raw] = '%8s' % eid + ''.join('%8s' % n for n in n9)
            nflip += 1
    open(src, 'w').write('\n'.join(lines) + '\n')
    # verify
    _, _, _, _, xy2, conn2 = parse_sg(src)
    neg = sum(1 for c in conn2 if signed_area(xy2[[n - 1 for n in c]]) < 0)
    print('s00: flipped %d inverted triangle(s) -> %d negative remaining' % (nflip, neg))


# ---------------------------------------------------------------- s02,s50: yaml -> .sg
def frame_to_angles(o):
    e1 = np.array(o[0:3]); e3 = np.array(o[6:9])
    th1 = np.degrees(np.arctan2(-e3[0], e3[1]))
    c1, s1 = np.cos(np.radians(th1)), np.sin(np.radians(th1))
    c3 = e1[2]; s3 = e1[0] * c1 + e1[1] * s1
    th3 = np.degrees(np.arctan2(s3, c3))
    return th1, th3


def convert(yaml_path, sg_path):
    d = yaml.safe_load(open(yaml_path))
    xy = np.array([_row(n)[:2] for n in d['nodes']])
    conn = [[int(round(v)) for v in _row(e)] for e in d['elements']]      # 1-based node ids
    ori = [_row(o) for o in d['elementOrientations']]
    # materials (ordered) + name->matid
    mats = []
    for m in d['materials']:
        E = [float(v) for v in m['E']]; G = [float(v) for v in m['G']]; nu = [float(v) for v in m['nu']]
        mats.append({'name': m['name'], 'E': E, 'G': G, 'nu': nu, 'rho': float(m['rho'])})
    mid = {m['name']: i + 1 for i, m in enumerate(mats)}
    # element -> material via sets
    sets = d['sets']['element'] if isinstance(d['sets'], dict) else d['sets']
    emat = np.zeros(len(conn), int)
    for s in sets:
        for lab in s['labels']:
            emat[int(lab) - 1] = mid[s['name']]
    # per-element theta1/theta3 ; group by (matid, round theta3)
    th1 = np.zeros(len(conn)); th3 = np.zeros(len(conn))
    for e in range(len(conn)):
        th1[e], th3[e] = frame_to_angles(ori[e])
    groups = {}                                                          # (matid, th3r) -> group_id
    egroup = np.zeros(len(conn), int)
    gdef = []
    for e in range(len(conn)):
        key = (emat[e], round(th3[e], 2))
        if key not in groups:
            groups[key] = len(gdef) + 1
            gdef.append((emat[e], round(th3[e], 6)))
        egroup[e] = groups[key]
    # ensure CCW (positive area) connectivity
    conn2 = []
    for c in conn:
        p = xy[[n - 1 for n in c]]
        conn2.append(c if signed_area(p) >= 0 else [c[0]] + c[1:][::-1])
    write_sg(sg_path, xy, conn2, egroup, th1, gdef, mats)
    neg = sum(1 for c in conn2 if signed_area(xy[[n - 1 for n in c]]) < 0)
    print('%s: %d nodes %d quads %d materials %d layer-groups  neg=%d  bbox_x=[%.3f,%.3f]'
          % (os.path.basename(sg_path), len(xy), len(conn2), len(mats), len(gdef), neg,
             xy[:, 0].min(), xy[:, 0].max()))


if __name__ == '__main__':
    flip_sg00()
    convert(os.path.join(FB, 'iea_s02_solid.yaml'), os.path.join(SGV, 'iea_s02.sg'))
    convert(os.path.join(FB, 'iea_s50_solid.yaml'), os.path.join(SGV, 'iea_s50.sg'))
    print('\nsg_v201 station count:', len([f for f in os.listdir(SGV) if f.endswith('.sg')]))
