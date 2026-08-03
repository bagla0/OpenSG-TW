#!/usr/bin/env python
"""Validate VABS-center / sectional formulas against the r0020 .K, then
against OpenSG solid stiffness.  Pure numpy.  0-idx VABS order:
  0=extension, 1=shear2, 2=shear3, 3=torsion, 4=bending2, 5=bending3.
"""
import numpy as np, re, os

BASE = os.path.dirname(os.path.abspath(__file__))
KFILE   = os.path.join(BASE, "sg",      "iea_r0020.sg.K")
MESH    = os.path.join(BASE, "2d_yaml", "iea_r0020_solid.yaml")
JAXOUT  = os.path.join(BASE, "out", "OpenSG_JAX_Solid",     "iea_r0020_OpenSG_JAX_Solid.out")
FENOUT  = os.path.join(BASE, "out", "OpenSG_FEniCSx_Solid", "iea_r0020_OpenSG_FEniCSx_Solid.out")

np.set_printoptions(precision=6, suppress=False, linewidth=160)

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def read_block_after(lines, header_pat, nrows=6, ncols=6):
    """Return the nrows x ncols float matrix whose first row is the first
    all-numeric line at or after a line matching header_pat."""
    for i, ln in enumerate(lines):
        if re.search(header_pat, ln):
            j = i + 1
            rows = []
            while len(rows) < nrows and j < len(lines):
                nums = re.findall(r'[-+]?\d+\.\d+[Ee][-+]?\d+', lines[j])
                if len(nums) >= ncols:
                    rows.append([float(x) for x in nums[:ncols]])
                j += 1
            return np.array(rows)
    raise RuntimeError("header not found: " + header_pat)

def scalar_after(lines, pat):
    for ln in lines:
        m = re.search(pat, ln)
        if m:
            nums = re.findall(r'[-+]?\d+\.\d+[Ee][-+]?\d+', ln)
            if nums:
                return float(nums[-1])
    return None

def Tf(a, b):
    """6x6 force/moment transform: F_new = Tf F_old when the reference
    point is moved to section position p=(x2,x3)=(a,b).
    M1'=M1+b F2 -a F3 ; M2'=M2 -b F1 ; M3'=M3 +a F1 ; forces unchanged."""
    T = np.eye(6)
    T[3,1] =  b;  T[3,2] = -a      # torsion row  (idx3)
    T[4,0] = -b                     # bending2 row (idx4)
    T[5,0] =  a                     # bending3 row (idx5)
    return T

def princ_2x2(p, q, r):
    """min,max eigenvalues and angle(deg,[0,180)) of the min-eigvec of
    [[p,q],[q,r]].  Axis measured atan2(v3,v2) about +x1."""
    tr = p + r; dsc = np.sqrt(((p - r)/2.0)**2 + q*q)
    lmin = tr/2.0 - dsc; lmax = tr/2.0 + dsc
    # eigenvector for lmin
    w, V = np.linalg.eigh(np.array([[p, q],[q, r]]))
    v = V[:, 0]                     # column for smallest eigenvalue
    ang = np.degrees(np.arctan2(v[1], v[0])) % 180.0
    return lmin, lmax, ang

def classical(K):
    """4x4 classical stiffness/compliance from 6x6 K, idx=[0,3,4,5]."""
    idx = [0,3,4,5]
    S = np.linalg.inv(K)
    Cc = S[np.ix_(idx, idx)]        # classical compliance
    Cs = np.linalg.inv(Cc)          # classical stiffness
    return Cs, Cc

# ----------------------------------------------------------------------
# load the .K
# ----------------------------------------------------------------------
kl = open(KFILE).read().splitlines()
K = read_block_after(kl, r'Timoshenko Stiffness Matrix')
M = read_block_after(kl, r'6X6 Mass Matrix\b')      # first (at origin)
S = np.linalg.inv(K)

# printed reference scalars/blocks
ref = dict(
    mass_center = np.array([scalar_after([kl[14]], r'.') ]),  # placeholder
)
# grab printed values directly
def grab_pair(lines, header):
    for i, ln in enumerate(lines):
        if re.search(header, ln):
            for j in range(i+1, min(i+6, len(lines))):
                nums = re.findall(r'[-+]?\d+\.\d+[Ee][-+]?\d+', lines[j])
                if len(nums) >= 2:
                    return float(nums[0]), float(nums[1])
    return None

pr = {}
pr['mass_center']   = grab_pair(kl, r'The Mass Center of the Cross Section')
pr['geom_center']   = grab_pair(kl, r'The Geometric Center of the Cross Section')
pr['tension_center']= grab_pair(kl, r'The Tension Center of the Cross Section')
pr['shear_center']  = grab_pair(kl, r'The Shear Center of the Cross Section')
pr['mc_wrt_sc']     = grab_pair(kl, r'The Mass Center with respect to Shear Center')
pr['mu']    = scalar_after(kl, r'Mass per unit span')
pr['i11']   = scalar_after(kl, r'Mass moment of inertia i11')
pr['i22']   = scalar_after(kl, r'Principal mass moments of inertia i22')
pr['i33']   = scalar_after(kl, r'Principal mass moments of inertia i33')
pr['rg']    = scalar_after(kl, r'mass-weighted radius of gyration')
pr['area']  = scalar_after(kl, r'^\s*Area =')
pr['EA']    = scalar_after(kl, r'The extension stiffness EA')
pr['GJ']    = scalar_after(kl, r'The torsional stiffness GJ')
pr['EI22']  = scalar_after(kl, r'Principal bending stiffness EI22')
pr['EI33']  = scalar_after(kl, r'Principal bending stiffness EI33')
pr['GA22']  = scalar_after(kl, r'Principal shear stiffness GA22')
pr['GA33']  = scalar_after(kl, r'Principal shear stiffness GA33')
# angles are on the line AFTER the 'rotated ... by' text
def grab_angle(lines, tag):
    for i, ln in enumerate(lines):
        if 'rotated' in ln and tag in ln:
            m = re.search(r'([-+]?\d+\.\d+)\s+degrees', lines[i+1])
            if m: return float(m.group(1))
    return None
# fallback: numbers appear on next line
def grab_angle2(lines, header_pat):
    for i, ln in enumerate(lines):
        if re.search(header_pat, ln):
            for j in range(i, min(i+3, len(lines))):
                m = re.search(r'([-+]?\d+\.\d+)\s+degrees', lines[j])
                if m: return float(m.group(1))
    return None
pr['ang_I']  = grab_angle2(kl, r'principal inertial axes rotated')
pr['ang_EI'] = grab_angle2(kl, r'principal bending axes rotated')
pr['ang_GA'] = grab_angle2(kl, r'principal shear axes rotated')

pr['Mc']      = read_block_after(kl, r'6X6 Mass Matrix at the Mass Center')
pr['ClsStiff']= read_block_after(kl, r'Classical Stiffness Matrix \(1-extension', 4, 4)
pr['ClsComp'] = read_block_after(kl, r'Classical Compliance Matrix \(1-extension', 4, 4)
pr['ClsStiffSC']= read_block_after(kl, r'Classical Stiffness Matrix at Shear Center', 4, 4)
pr['ClsCompSC'] = read_block_after(kl, r'Classical Compliance Matrix at Shear Center', 4, 4)
pr['MassSC']  = read_block_after(kl, r'The Mass Matrix at Shear Center')
pr['ComplTimo']= read_block_after(kl, r'Timoshenko Compliance Matrix')
# shear-center mass properties
def grab_sc_massprops(lines):
    d = {}
    for i, ln in enumerate(lines):
        if 'The Mass Properties at Shear Center' in ln:
            seg = lines[i:i+12]
            d['mu']  = scalar_after(seg, r'Mass per unit span')
            d['i11'] = scalar_after(seg, r'Mass moment of inertia i11')
            d['i22'] = scalar_after(seg, r'Principal mass moments of inertia i22')
            d['i33'] = scalar_after(seg, r'Principal mass moments of inertia i33')
            d['rg']  = scalar_after(seg, r'radius of gyration')
            d['ang'] = grab_angle2(seg, r'principal inertial axes rotated')
    return d
pr['scmass'] = grab_sc_massprops(kl)

# ----------------------------------------------------------------------
# COMPUTE everything from K and M
# ----------------------------------------------------------------------
c = {}
mu = M[0,0]
c['mu'] = mu
# mass center
x2c = -M[0,5]/M[0,0]; x3c = M[0,4]/M[0,0]
c['mass_center'] = (x2c, x3c)
# mass at mass center
Mc = Tf(x2c, x3c) @ M @ Tf(x2c, x3c).T
c['Mc'] = Mc
c['i11'] = Mc[3,3]
i22, i33, angI = princ_2x2(Mc[4,4], Mc[4,5], Mc[5,5])
c['i22'], c['i33'], c['ang_I'] = i22, i33, angI
c['rg'] = np.sqrt(Mc[3,3]/mu)

# classical
Cs, Cc = classical(K)
c['ClsStiff'] = Cs; c['ClsComp'] = Cc
c['EA'] = Cs[0,0]
c['GJ'] = Cs[1,1]
# tension center
x2t = -K[0,5]/K[0,0]; x3t = K[0,4]/K[0,0]
c['tension_center'] = (x2t, x3t)
# principal bending at tension center (transform classical 4x4)
T4 = np.eye(4); T4[2,0] = -x3t; T4[3,0] = x2t
Csc4 = T4 @ Cs @ T4.T
EI22, EI33, angEI = princ_2x2(Csc4[2,2], Csc4[2,3], Csc4[3,3])
c['EI22'], c['EI33'], c['ang_EI'] = EI22, EI33, angEI
# shear center (from compliance)
x2s = -S[3,2]/S[3,3]; x3s = S[3,1]/S[3,3]
c['shear_center'] = (x2s, x3s)
# principal shear (invariant transverse-shear block)
GA22, GA33, angGA = princ_2x2(K[1,1], K[1,2], K[2,2])
c['GA22'], c['GA33'], c['ang_GA'] = GA22, GA33, angGA
# classical at shear center
Ksc = Tf(x2s, x3s) @ K @ Tf(x2s, x3s).T
Ssc = np.linalg.inv(Ksc)
idx = [0,3,4,5]
CcSC = Ssc[np.ix_(idx, idx)]; CsSC = np.linalg.inv(CcSC)
c['ClsStiffSC'] = CsSC; c['ClsCompSC'] = CcSC
# mass at shear center
Msc = Tf(x2s, x3s) @ M @ Tf(x2s, x3s).T
c['MassSC'] = Msc
c['mc_wrt_sc'] = (x2c - x2s, x3c - x3s)
# mass props at shear center
sc = {}
sc['mu'] = Msc[0,0]; sc['i11'] = Msc[3,3]
sc['i22'], sc['i33'], sc['ang'] = princ_2x2(Msc[4,4], Msc[4,5], Msc[5,5])
sc['rg'] = np.sqrt(Msc[3,3]/Msc[0,0])
c['scmass'] = sc
c['ComplTimo'] = S

# ----------------------------------------------------------------------
# MESH: area + geometric center
# ----------------------------------------------------------------------
def parse_mesh(path):
    nodes = []; elems = []
    with open(path) as f:
        mode = None
        for ln in f:
            s = ln.strip()
            if s == 'nodes:': mode='n'; continue
            if s == 'elements:': mode='e'; continue
            if s in ('sets:','elementOrientations:','materials:') or (s.endswith(':') and not s.startswith('-')):
                if mode in ('n','e'): mode=None
            if s.startswith('- [') and mode=='n':
                nodes.append([float(x) for x in s[3:-1].split()])
            elif s.startswith('- [') and mode=='e':
                elems.append([int(x) for x in s[3:-1].split()])
    return np.array(nodes), elems

nodes, elems = parse_mesh(MESH)
# detect index base
flat = [i for e in elems for i in e]
base = min(flat)
xy = nodes[:, :2]   # x2=col0, x3=col1
A = 0.0; Ax2 = 0.0; Ax3 = 0.0
for e in elems:
    p = xy[[i - base for i in e]]
    # shoelace (works for tri & quad, ordered)
    x = p[:,0]; y = p[:,1]
    ar = 0.5*np.sum(x*np.roll(y,-1) - np.roll(x,-1)*y)
    ar = abs(ar)
    cx = p[:,0].mean(); cy = p[:,1].mean()
    A += ar; Ax2 += ar*cx; Ax3 += ar*cy
c['area'] = A
c['geom_center'] = (Ax2/A, Ax3/A)

# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def pe(comp, refv):
    if refv is None: return None
    if abs(refv) < 1e-30: return abs(comp - refv)
    return 100.0*abs(comp - refv)/abs(refv)

def frob_pe(comp, refm):
    d = np.linalg.norm(comp - refm)
    n = np.linalg.norm(refm)
    return 100.0*d/n if n>0 else d

rows = []
def add(name, comp, refv, kind='scalar'):
    if kind=='scalar':
        e = pe(comp, refv)
        st = 'MATCH' if (e is not None and e<0.1) else ('CLOSE' if (e is not None and e<1.0) else 'MISMATCH')
        rows.append((name, f"{comp:.6g}", f"{refv:.6g}" if refv is not None else "N/A",
                     f"{e:.4g}" if e is not None else "N/A", st))
    elif kind=='mat':
        e = frob_pe(comp, refv)
        st = 'MATCH' if e<0.1 else ('CLOSE' if e<1.0 else 'MISMATCH')
        rows.append((name, "matrix", "matrix", f"{e:.4g}(Frob%)", st))

add('EA', c['EA'], pr['EA'])
add('GJ', c['GJ'], pr['GJ'])
add('EI22', c['EI22'], pr['EI22'])
add('EI33', c['EI33'], pr['EI33'])
add('GA22', c['GA22'], pr['GA22'])
add('GA33', c['GA33'], pr['GA33'])
add('mu', c['mu'], pr['mu'])
add('i11', c['i11'], pr['i11'])
add('i22', c['i22'], pr['i22'])
add('i33', c['i33'], pr['i33'])
add('rg', c['rg'], pr['rg'])
add('ang_inertial', c['ang_I'], pr['ang_I'])
add('ang_bending',  c['ang_EI'], pr['ang_EI'])
add('ang_shear',    c['ang_GA'], pr['ang_GA'])
add('mass_center_x2', c['mass_center'][0], pr['mass_center'][0])
add('mass_center_x3', c['mass_center'][1], pr['mass_center'][1])
add('tension_center_x2', c['tension_center'][0], pr['tension_center'][0])
add('tension_center_x3', c['tension_center'][1], pr['tension_center'][1])
add('shear_center_x2', c['shear_center'][0], pr['shear_center'][0])
add('shear_center_x3', c['shear_center'][1], pr['shear_center'][1])
add('mc_wrt_sc_x2', c['mc_wrt_sc'][0], pr['mc_wrt_sc'][0])
add('mc_wrt_sc_x3', c['mc_wrt_sc'][1], pr['mc_wrt_sc'][1])
add('area(mesh)', c['area'], pr['area'])
add('geom_center_x2(mesh)', c['geom_center'][0], pr['geom_center'][0])
add('geom_center_x3(mesh)', c['geom_center'][1], pr['geom_center'][1])
# matrices
add('Mass@MassCenter 6x6', c['Mc'], pr['Mc'], 'mat')
add('Classical Stiff 4x4', c['ClsStiff'], pr['ClsStiff'], 'mat')
add('Classical Compl 4x4', c['ClsComp'], pr['ClsComp'], 'mat')
add('Classical Stiff@SC 4x4', c['ClsStiffSC'], pr['ClsStiffSC'], 'mat')
add('Classical Compl@SC 4x4', c['ClsCompSC'], pr['ClsCompSC'], 'mat')
add('Mass@SC 6x6', c['MassSC'], pr['MassSC'], 'mat')
add('Timoshenko Compl 6x6', c['ComplTimo'], pr['ComplTimo'], 'mat')
# sc mass props scalars
add('SC i11', c['scmass']['i11'], pr['scmass']['i11'])
add('SC i22', c['scmass']['i22'], pr['scmass']['i22'])
add('SC i33', c['scmass']['i33'], pr['scmass']['i33'])
add('SC rg',  c['scmass']['rg'],  pr['scmass']['rg'])
add('SC ang', c['scmass']['ang'], pr['scmass']['ang'])

print("="*100)
print("PART (a)+(b):  formulas from r0020 .K's own Timo 6x6 + Mass 6x6  (mesh for area/geom-center)")
print("="*100)
print(f"{'quantity':30s} {'computed':>16s} {'vabs_K':>16s} {'%err':>14s}  status")
print("-"*100)
for nm, cc, rr, ee, st in rows:
    print(f"{nm:30s} {cc:>16s} {rr:>16s} {ee:>14s}  {st}")

# ----------------------------------------------------------------------
# PART (c): cross-check vs OpenSG solid
# ----------------------------------------------------------------------
def load_opensg(path):
    ls = open(path).read().splitlines()
    return read_block_after(ls, r'^Stiffness')

print("\n" + "="*100)
print("PART (c): cross-check EA/GJ/EI/GA + centers from OpenSG solid 6x6 vs .K")
print("="*100)
for tag, path in [('JAX', JAXOUT), ('FEniCSx', FENOUT)]:
    if not os.path.exists(path):
        print(f"[{tag}] missing"); continue
    Ko = load_opensg(path)
    So = np.linalg.inv(Ko)
    Cso, _ = classical(Ko)
    EAo = Cso[0,0]; GJo = Cso[1,1]
    x2to = -Ko[0,5]/Ko[0,0]; x3to = Ko[0,4]/Ko[0,0]
    x2so = -So[3,2]/So[3,3]; x3so = So[3,1]/So[3,3]
    # principal bending (invariants) at tension center
    T4o = np.eye(4); T4o[2,0] = -x3to; T4o[3,0] = x2to
    Csc4o = T4o @ Cso @ T4o.T
    EI22o, EI33o, _ = princ_2x2(Csc4o[2,2], Csc4o[2,3], Csc4o[3,3])
    GA22o, GA33o, _ = princ_2x2(Ko[1,1], Ko[1,2], Ko[2,2])
    print(f"\n--- OpenSG {tag} solid ---")
    print(f"  {'qty':10s} {'OpenSG':>16s} {'.K':>16s} {'%err':>10s}")
    for nm, vo, vk in [('EA',EAo,pr['EA']),('GJ',GJo,pr['GJ']),
                       ('EI22',EI22o,pr['EI22']),('EI33',EI33o,pr['EI33']),
                       ('GA22',GA22o,pr['GA22']),('GA33',GA33o,pr['GA33']),
                       ('tc_x2',x2to,pr['tension_center'][0]),('tc_x3',x3to,pr['tension_center'][1]),
                       ('sc_x2',x2so,pr['shear_center'][0]),('sc_x3',x3so,pr['shear_center'][1])]:
        e = pe(vo, vk)
        print(f"  {nm:10s} {vo:>16.6g} {vk:>16.6g} {e:>9.3g}%")

# ----------------------------------------------------------------------
# overall verdict
# ----------------------------------------------------------------------
crit = ['EA','GJ','EI22','EI33','GA22','GA33',
        'mass_center_x2','mass_center_x3','tension_center_x2','tension_center_x3',
        'shear_center_x2','shear_center_x3']
worst = 0.0
for nm, cc, rr, ee, st in rows:
    if nm in crit and ee not in ('N/A',):
        try: worst = max(worst, float(ee))
        except: pass
print("\n" + "="*100)
print(f"Worst %err among stiffness-derived centers/principal values: {worst:.4g}%  ->  "
      f"{'ALL <1% PASS' if worst<1.0 else 'SOME >1%'}")
print("="*100)
