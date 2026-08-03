'''sectional_props.py -- VABS-.K sectional properties (centers / classical / principal / mass props)
from a Timoshenko 6x6 stiffness K + 6x6 mass M, plus the two mass builders:

  * shell 6x6 mass  = FAITHFUL get_mass_shell (per-layup through-thickness moments about the mid-
    surface integrated over the contour, through-thickness = global x3).  This is the version that
    matches the VABS .K 6x6 mass (incl. the M56 sign); the wall-normal 'corrected' one flips M56.
  * solid 6x6 mass  = area integral over the 2-D mesh with M56 = -i23 (i23=INT x2 x3 rho dA), the
    VABS sign (get_mass_solid's +i23 is wrong).

All formulas validated in validate_vabs_center_formulas.py against iea_r0020.sg.K (<1%).
VABS/OpenSG 0-idx order: 0=ext, 1=shear2, 2=shear3, 3=torsion, 4=bending2, 5=bending3.
Everything is referenced to the x1 reference axis (the yamls are shifted before homogenizing).
'''
import os
import numpy as np
import yaml

# ------------------------------------------------------------------ K/M algebra
def Tf(a, b):
    '''6x6 force/moment transform when the reference point moves to section (x2,x3)=(a,b).'''
    T = np.eye(6)
    T[3, 1] = b;  T[3, 2] = -a
    T[4, 0] = -b
    T[5, 0] = a
    return T


def princ_2x2(p, q, r):
    '''min,max eigenvalue and angle(deg,[0,180)) of the min-eigvec of [[p,q],[q,r]] about +x1.'''
    tr = p + r; dsc = np.sqrt(((p - r) / 2.0) ** 2 + q * q)
    lmin = tr / 2.0 - dsc; lmax = tr / 2.0 + dsc
    w, V = np.linalg.eigh(np.array([[p, q], [q, r]]))
    v = V[:, 0]
    ang = np.degrees(np.arctan2(v[1], v[0])) % 180.0
    return lmin, lmax, ang


def classical(K):
    idx = [0, 3, 4, 5]
    S = np.linalg.inv(K)
    Cc = S[np.ix_(idx, idx)]
    Cs = np.linalg.inv(Cc)
    return Cs, Cc


def compute_props(K, M, area, geom_center):
    '''Return a dict with every VABS-.K sectional quantity computed from K, M, mesh area+geom center.'''
    S = np.linalg.inv(K)
    c = {}
    c['K'] = K; c['S'] = S; c['M'] = M
    mu = M[0, 0]
    # mass center + mass at mass center + principal inertia
    x2c = -M[0, 5] / M[0, 0]; x3c = M[0, 4] / M[0, 0]
    c['mass_center'] = (x2c, x3c)
    Mc = Tf(x2c, x3c) @ M @ Tf(x2c, x3c).T
    c['Mc'] = Mc
    c['mu'] = mu; c['i11'] = Mc[3, 3]
    i22, i33, angI = princ_2x2(Mc[4, 4], Mc[4, 5], Mc[5, 5])
    c['i22'], c['i33'], c['ang_I'] = i22, i33, angI
    c['rg'] = np.sqrt(Mc[3, 3] / mu)
    # geometry
    c['area'] = area; c['geom_center'] = geom_center
    # classical 4x4 + tension center + principal bending
    Cs, Cc = classical(K)
    c['ClsStiff'] = Cs; c['ClsComp'] = Cc
    c['EA'] = Cs[0, 0]; c['GJ'] = Cs[1, 1]
    x2t = -K[0, 5] / K[0, 0]; x3t = K[0, 4] / K[0, 0]
    c['tension_center'] = (x2t, x3t)
    T4 = np.eye(4); T4[2, 0] = -x3t; T4[3, 0] = x2t
    Csc4 = T4 @ Cs @ T4.T
    EI22, EI33, angEI = princ_2x2(Csc4[2, 2], Csc4[2, 3], Csc4[3, 3])
    c['EI22'], c['EI33'], c['ang_EI'] = EI22, EI33, angEI
    # shear center + principal shear + classical/mass at SC
    x2s = -S[3, 2] / S[3, 3]; x3s = S[3, 1] / S[3, 3]
    c['shear_center'] = (x2s, x3s)
    GA22, GA33, angGA = princ_2x2(K[1, 1], K[1, 2], K[2, 2])
    c['GA22'], c['GA33'], c['ang_GA'] = GA22, GA33, angGA
    Ksc = Tf(x2s, x3s) @ K @ Tf(x2s, x3s).T
    Ssc = np.linalg.inv(Ksc)
    idx = [0, 3, 4, 5]
    CcSC = Ssc[np.ix_(idx, idx)]; CsSC = np.linalg.inv(CcSC)
    c['ClsStiffSC'] = CsSC; c['ClsCompSC'] = CcSC
    Msc = Tf(x2s, x3s) @ M @ Tf(x2s, x3s).T
    c['MassSC'] = Msc
    c['mc_wrt_sc'] = (x2c - x2s, x3c - x3s)
    sc = {}
    sc['mu'] = Msc[0, 0]; sc['i11'] = Msc[3, 3]
    sc['i22'], sc['i33'], sc['ang'] = princ_2x2(Msc[4, 4], Msc[4, 5], Msc[5, 5])
    sc['rg'] = np.sqrt(Msc[3, 3] / Msc[0, 0])
    c['scmass'] = sc
    return c


# ------------------------------------------------------------------ yaml parsing
def _row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(',', ' ').split()]
    return [float(v) for v in r]


# ------------------------------------------------------------------ shell mass (FAITHFUL)
def shell_mass_and_geom(path):
    '''FAITHFUL get_mass_shell 6x6 + material area/geom-center (thickness-weighted contour).'''
    d = yaml.safe_load(open(path))
    rx = np.array([_row(r)[:3] for r in d['nodes']], float)
    cells = np.array([[int(v) for v in _row(e)] for e in d['elements']], int)
    if cells.min() == 1:
        cells = cells - 1
    sections = d['sections']; materials = d['materials']
    rho_by = {m['name']: float(m['density']) for m in materials}
    set2sec = {s['elementSet']: i for i, s in enumerate(sections)}
    rsub = np.zeros(len(cells), int)
    for grp in d['sets']['element']:
        si = set2sec[grp['name']]
        for lab in grp['labels']:
            rsub[int(lab) - 1] = si
    mom = np.zeros((len(sections), 3))          # (mu, mx3, i22) about mid-surface
    Ttot = np.zeros(len(sections))              # total layup thickness
    for si, sec in enumerate(sections):
        layup = sec['layup']
        th = np.array([float(p[1]) for p in layup])
        rho = np.array([rho_by[p[0]] for p in layup])
        T = th.sum(); Ttot[si] = T
        z_bot = np.concatenate([[0.0], np.cumsum(th)])[:-1]
        z_mid = z_bot + 0.5 * th - 0.5 * T
        mom[si] = (np.sum(rho * th),
                   np.sum(rho * th * z_mid),
                   np.sum(rho * (th * z_mid ** 2 + th ** 3 / 12.0)))
    cross = [0, 1]
    a = rx[cells[:, 0]][:, cross]; b = rx[cells[:, 1]][:, cross]
    ds = np.linalg.norm(b - a, axis=1)
    g = 1.0 / np.sqrt(3.0); xi = np.array([-g, g])
    xg = 0.5 * (1 - xi)[None, :, None] * a[:, None, :] + 0.5 * (1 + xi)[None, :, None] * b[:, None, :]
    x2 = xg[:, :, 0]; x3 = xg[:, :, 1]
    mu = mom[rsub, 0][:, None]; mx3 = mom[rsub, 1][:, None]; i22 = mom[rsub, 2][:, None]
    w = ds[:, None] * 0.5

    def I(f):
        return float(np.sum(np.broadcast_to(f, x2.shape) * w))
    M11 = I(mu)
    M15 = I(mx3 + x3 * mu)
    M16 = I(-x2 * mu)
    M44 = I(i22 + 2 * x3 * mx3 + mu * x2 ** 2 + x3 ** 2 * mu)
    M55 = I(i22 + 2 * x3 * mx3 + x3 ** 2 * mu)
    M66 = I(mu * x2 ** 2)
    M56 = I(-x2 * (mx3 + x3 * mu))
    M = np.array([
        (M11, 0, 0, 0, M15, M16),
        (0, M11, 0, -M15, 0, 0),
        (0, 0, M11, -M16, 0, 0),
        (0, -M15, -M16, M44, 0, 0),
        (M15, 0, 0, 0, M55, M56),
        (M16, 0, 0, 0, M56, M66)], float)
    # material area + thickness-weighted geometric center (shell analog of VABS area)
    Te = Ttot[rsub]
    xmid = 0.5 * (a + b)
    dA = ds * Te
    area = float(dA.sum())
    gc = (float(np.sum(dA * xmid[:, 0]) / area), float(np.sum(dA * xmid[:, 1]) / area))
    return M, area, gc


# ------------------------------------------------------------------ solid mass (M56=-i23)
def solid_mass_and_geom(path, windio=None):
    d = yaml.safe_load(open(path))
    nd = np.array([_row(r)[:2] for r in d['nodes']], float)
    el = [[int(v) for v in _row(e)] for e in d['elements']]
    dens = {}
    for m in d.get('materials', []):
        rho = m.get('density', m.get('rho'))
        if isinstance(rho, (list, tuple)):
            rho = rho[0]
        if rho is not None:
            dens[m['name']] = float(rho)
    if windio and os.path.exists(windio):
        wd = yaml.safe_load(open(windio))
        for m in wd.get('materials', []):
            dens.setdefault(m['name'], float(m.get('rho', 0.0)))
    set2mat = {}
    for s in d.get('sections', []):
        set2mat[s['elementSet']] = s.get('material', s.get('name'))
    el_mat = [None] * len(el)
    for grp in d['sets']['element']:
        mat = set2mat.get(grp['name'], grp['name'])
        for lab in grp['labels']:
            el_mat[int(lab) - 1] = mat
    mu = xm2n = xm3n = i22 = i33 = i23 = 0.0
    A = Ax2 = Ax3 = 0.0
    miss = set()
    for k, e in enumerate(el):
        idx = [i - 1 for i in e]
        P = nd[idx]
        if len(idx) == 3:
            ar = 0.5 * abs((P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1]) -
                           (P[2, 0] - P[0, 0]) * (P[1, 1] - P[0, 1]))
        else:
            x, y = P[:, 0], P[:, 1]
            ar = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
        cc = P.mean(0)
        rho = dens.get(el_mat[k])
        if rho is None:
            miss.add(el_mat[k]); rho = 0.0
        m = rho * ar
        mu += m; xm2n += cc[0] * m; xm3n += cc[1] * m
        i22 += cc[1] ** 2 * m; i33 += cc[0] ** 2 * m; i23 += cc[0] * cc[1] * m
        A += ar; Ax2 += ar * cc[0]; Ax3 += ar * cc[1]
    xm2, xm3 = xm2n / mu, xm3n / mu
    M = np.array([[mu, 0, 0, 0, mu * xm3, -mu * xm2],
                  [0, mu, 0, -mu * xm3, 0, 0],
                  [0, 0, mu, mu * xm2, 0, 0],
                  [0, -mu * xm3, mu * xm2, i22 + i33, 0, 0],
                  [mu * xm3, 0, 0, 0, i22, -i23],       # M56 = -i23 (VABS sign)
                  [-mu * xm2, 0, 0, 0, -i23, i33]], float)
    return M, float(A), (float(Ax2 / A), float(Ax3 / A)), miss
