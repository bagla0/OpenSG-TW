#!/usr/bin/env python
'''
precheck_prevabs.py  --  preconditioning check for a generated PreVABS cross-section XML.

PreVABS builds a 2-D SOLID cross-section by reading a standard Selig/UIUC airfoil `.dat`
(the OML contour) and OFFSETTING each laminate INWARD from it.  The airfoil itself is fine;
what fails is the XML/geometry WE generate around it (division points, baselines, layup
thicknesses vs the local contour resolution).  This module validates the generated `.xml`
(+ its `.dat` and `materials.xml`) against everything PreVABS needs, WITHOUT touching the
airfoil, and tags each finding:

    severity  : 'info' | 'warn' | 'fatal'
    fixable   : True  -> a shape-preserving preprocessing step can resolve it (see condition_prevabs.py)
                False -> PreVABS genuinely cannot handle it -> route to the pyNuMAD-style fallback

The single most important check is OFFSET FEASIBILITY: PreVABS's own diagnostic
(geo_diagnostics.cpp:checkOffsetDistanceVsShortestEdge) fails when a laminate's inward
offset distance is much larger than the shortest baseline segment it is offset from.  That is
exactly what kills s02 (thick root laminate over short LE-region segments, ratio ~43x) and
s50 (tip trailing-edge sliver, ratio ~3700x).

USAGE
    python precheck_prevabs.py path/to/iea_s02.xml
    python precheck_prevabs.py 'xml/*.xml'            # batch (quote the glob)
    python precheck_prevabs.py iea_s02.xml --json     # machine-readable

Exit code: 0 clean, 1 warnings only, 2 has fatal (fixable), 3 has fatal-unfixable (needs fallback).

Importable:
    from precheck_prevabs import precheck
    report = precheck('iea_s02.xml')      # -> dict(issues=[...], verdict=..., ...)
'''
import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

# offset/shortest-segment ratio thresholds (mirror PreVABS geo_diagnostics intent)
RATIO_WARN = 2.0        # numerically fragile
RATIO_FATAL = 8.0       # offset construction will very likely collapse
MIN_PHYS_SEG = 1e-3     # m; baseline segments shorter than this (physical) are trouble
TE_MIN_GAP = 5e-4       # m; trailing-edge thinner than this is a sliver
COINC_EPS = 1e-6        # normalized-x coincidence tolerance for dividing points


class Issue(dict):
    def __init__(self, severity, code, message, fixable, fix=''):
        super().__init__(severity=severity, code=code, message=message, fixable=fixable, fix=fix)


# --------------------------------------------------------------------- parsers
def parse_dat(path):
    '''Selig/UIUC .dat -> (name, Nx2 array). First line is a name/header unless it is 2 floats.'''
    with open(path, 'rb') as f:
        raw = f.read().decode('latin-1')
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    name = ''
    start = 0
    tok0 = lines[0].split()
    try:
        float(tok0[0]); float(tok0[1])          # first line already numeric -> no header
    except (ValueError, IndexError):
        name = lines[0]; start = 1
    pts = []
    for ln in lines[start:]:
        t = ln.split()
        if len(t) >= 2:
            pts.append((float(t[0]), float(t[1])))
    return name, np.array(pts, float)


_MAT_CACHE = {}


def parse_materials(path):
    '''materials.xml -> {lamina_name: thickness}. Cached by (path, mtime): ONE shared
    materials.xml serves all 51 stations, so it is parsed once, not per station.'''
    if not os.path.exists(path):
        return {}
    key = (path, os.path.getmtime(path))
    hit = _MAT_CACHE.get(key)
    if hit is not None:
        return hit
    thk = {}
    root = ET.parse(path).getroot()
    for lam in root.iter('lamina'):
        nm = lam.get('name')
        t = lam.findtext('thickness')
        if nm and t:
            thk[nm] = float(t)
    _MAT_CACHE[key] = thk
    return thk


def parse_xml(path):
    root = ET.parse(path).getroot()
    g = root.find('general')
    scale = float(g.findtext('scale')) if g is not None and g.findtext('scale') else None
    mesh = float(g.findtext('mesh_size')) if g is not None and g.findtext('mesh_size') else None
    normalize = None
    if g is not None and g.findtext('normalize'):
        normalize = g.findtext('normalize')
    bl = root.find('baselines')
    dat_ref, points, lines, webs = None, {}, {}, []
    if bl is not None:
        for e in bl:
            if e.tag == 'line' and e.get('type') == 'airfoil':
                p = e.find('points')
                dat_ref = (p.text or '').strip() if p is not None else None
            elif e.tag == 'point' and e.get('on'):          # dividing point on airfoil
                points[e.get('name')] = dict(side=e.get('which'), by=e.get('by'), x=float(e.text))
            elif e.tag == 'point':                          # web point "x y"
                xy = [float(v) for v in (e.text or '').split()]
                webs.append(dict(name=e.get('name'), xy=xy))
            elif e.tag == 'line' and e.get('name', '').startswith('bl'):
                pe = e.find('points')
                if pe is not None and ':' in (pe.text or ''):
                    a, b = pe.text.split(':')
                    lines[e.get('name')] = (a.strip(), b.strip())
    # layups -> total thickness (sum lamina_thk * count)
    layups = {}
    lu = root.find('layups')
    if lu is not None:
        for L in lu.findall('layup'):
            layers = []
            for lay in L.findall('layer'):
                lam = lay.get('lamina')
                cnt = 1
                txt = (lay.text or '').strip()
                if ':' in txt:
                    cnt = int(txt.split(':')[1])
                layers.append((lam, cnt))
            layups[L.get('name')] = layers
    # segment -> layup
    seg_layup = {}
    for comp in root.iter('component'):
        for sg in comp.findall('segment'):
            baseline = sg.findtext('baseline')
            layup = sg.findtext('layup')
            seg_layup[baseline] = layup
    return dict(scale=scale, mesh=mesh, normalize=normalize, dat_ref=dat_ref,
                points=points, lines=lines, webs=webs, layups=layups, seg_layup=seg_layup, root=root)


# --------------------------------------------------------------- geometry help
def arclen(pts):
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def seg_lengths(pts):
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def layup_thickness(layers, lam_thk):
    return sum(lam_thk.get(lam, 0.0) * cnt for lam, cnt in layers)


# --- airfoil-type classifier -> recommended pyNuMAD-quad (layered-sweep) params -----
# The near-circular THICK ROOT (thickness/chord -> 1, e.g. s00/s01) inverts the
# structured layered-quad loft at the default resolution; it needs a finer skin
# arc size, MORE through-thickness layers, and FEWER web-band subdivisions.
# Prescribing this up front (instead of discovering it via a slow retry ladder)
# lets the solid-mesh generator succeed FIRST TRY, so it is the fast path.
# Calibrated on IEA-22: t/c is 1.00 at s00/s01, 0.93 at s02 (default OK), -> 0.21 tip.
ROOT_TC = 0.95     # thickness/chord at/above which a section is a near-circular thick root


def recommend_mesh_params(dat):
    '''From the airfoil .dat, classify the section and return the recommended
    layered-quad mesh params (mesh_size, nr through-thickness layers, nw web-band
    subdivisions) for to_solid_hex.  x is normalized in [0,1] so chord=1 and
    thickness/chord = span of the y column.'''
    tc = float(dat[:, 1].max() - dat[:, 1].min())
    if tc >= ROOT_TC:
        return dict(airfoil_kind='thick_circular_root', thickness_ratio=round(tc, 4),
                    mesh_size=0.015, nr=6, nw=2)
    return dict(airfoil_kind='airfoil', thickness_ratio=round(tc, 4),
                mesh_size=0.02, nr=4, nw=3)


# ----------------------------------------------------------------- main check
def precheck(xml_path):
    issues = []
    d = os.path.dirname(os.path.abspath(xml_path))
    X = parse_xml(xml_path)
    scale = X['scale'] or 1.0

    # --- .dat ---------------------------------------------------------------
    dat_path = os.path.join(d, X['dat_ref']) if X['dat_ref'] else None
    dat = None
    if not dat_path or not os.path.exists(dat_path):
        issues.append(Issue('fatal', 'dat_missing', 'airfoil .dat referenced by ln_af not found: %s' % X['dat_ref'],
                            False, 'regenerate the .dat'))
    else:
        name, dat = parse_dat(dat_path)
        if len(dat) < 20:
            issues.append(Issue('fatal', 'dat_too_few', '.dat has only %d points' % len(dat), False, 'regenerate'))
        xs = dat[:, 0]
        if xs.min() < -1e-6 or xs.max() > 1.0 + 1e-6:
            issues.append(Issue('warn', 'dat_not_normalized',
                                '.dat x-range [%.4f, %.4f] not in [0,1] (PreVABS scales by <scale>)' % (xs.min(), xs.max()),
                                True, 'normalize x to [0,1] (shape preserved; scale=chord restores size)'))
        # Selig ordering: should start near TE (x~1), reach LE (x~0), return to TE
        if not (xs[0] > 0.5 and xs[np.argmin(xs)] < 0.1):
            issues.append(Issue('warn', 'dat_order',
                                'point order does not look Selig (start x=%.3f, min x=%.3f)' % (xs[0], xs.min()),
                                True, 're-order to Selig TE->LE->TE'))
        # coincident consecutive .dat points
        segL = seg_lengths(dat)
        ncoinc = int((segL < 1e-9).sum())
        if ncoinc:
            issues.append(Issue('warn', 'dat_coincident_pts', '%d coincident consecutive .dat points' % ncoinc,
                                True, 'drop duplicate points / uniform arc-length resample'))
        # shortest physical baseline segment
        phys = segL * scale
        nz = phys[phys > 1e-12]
        min_phys = float(nz.min()) if len(nz) else 0.0
        if min_phys < MIN_PHYS_SEG:
            issues.append(Issue('warn', 'dat_short_segments',
                                'shortest .dat segment = %.3g m (< %.3g m); makes inward offset fragile'
                                % (min_phys, MIN_PHYS_SEG),
                                True, 'uniform arc-length resample of the .dat (kills sub-mm segments, shape preserved)'))
        # trailing-edge gap (first vs last point = TE upper vs lower)
        te_gap = float(np.linalg.norm(dat[0] - dat[-1]) * scale)
        if te_gap < TE_MIN_GAP:
            issues.append(Issue('warn', 'te_sliver',
                                'trailing-edge gap = %.3g m (< %.3g m): near-zero-thickness sliver' % (te_gap, TE_MIN_GAP),
                                True, 'open a minimum TE gap (moves TE points a fraction of chord; airfoil interior unchanged)'))

    # --- XML general --------------------------------------------------------
    if not X['scale'] or X['scale'] <= 0:
        issues.append(Issue('fatal', 'bad_scale', '<scale> missing or <= 0', False, 'set <scale> = chord'))
    if not X['mesh'] or X['mesh'] <= 0:
        issues.append(Issue('fatal', 'bad_mesh', '<mesh_size> missing or <= 0', False, 'set a positive <mesh_size>'))

    # --- dividing points: per side strictly monotone & non-coincident -------
    for sidename in ('top', 'bottom'):
        pts = sorted(((nm, p['x']) for nm, p in X['points'].items() if p.get('side') == sidename),
                     key=lambda kv: kv[1])
        for (na, xa), (nb, xb) in zip(pts, pts[1:]):
            if abs(xa - xb) <= COINC_EPS:
                issues.append(Issue('fatal', 'coincident_div',
                                    'dividing points %s,%s coincide on %s side at x=%.6f -> zero-length baseline'
                                    % (na, nb, sidename, xa),
                                    True, 'spread the coincident dividing points apart (e.g. LE_XMIN clamp collision)'))

    # --- zero-length baselines ---------------------------------------------
    for bl, (a, b) in X['lines'].items():
        pa, pb = X['points'].get(a), X['points'].get(b)
        if pa and pb and pa.get('side') == pb.get('side') and abs(pa['x'] - pb['x']) <= COINC_EPS:
            issues.append(Issue('fatal', 'zero_baseline',
                                'baseline %s (%s:%s) has ~zero length' % (bl, a, b),
                                True, 'remove/merge the degenerate baseline (spread its endpoints)'))

    # --- OFFSET FEASIBILITY (the s02/s50 predictor) -------------------------
    lam_thk = parse_materials(os.path.join(d, 'materials.xml'))
    max_off = 0.0
    max_off_layup = None
    for lu, layers in X['layups'].items():
        t = layup_thickness(layers, lam_thk)          # physical metres
        if t > max_off:
            max_off, max_off_layup = t, lu
    if dat is not None and max_off > 0:
        phys = seg_lengths(dat) * scale
        nz = phys[phys > 1e-12]
        min_phys = float(nz.min()) if len(nz) else 0.0
        # worst-case global ratio (matches geo_diagnostics: offset dist vs shortest base segment)
        if min_phys > 0:
            ratio = max_off / min_phys
            half_min_dim = 0.5 * float((dat[:, 1].max() - dat[:, 1].min()) * scale)  # ~ local room proxy
            if max_off >= half_min_dim:
                issues.append(Issue('fatal', 'offset_exceeds_section',
                                    'thickest layup %s = %.3g m offsets deeper than the section half-thickness %.3g m'
                                    % (max_off_layup, max_off, half_min_dim),
                                    False, 'geometry cannot hold the laminate; use the fallback layered mesher'))
            elif ratio >= RATIO_FATAL:
                issues.append(Issue('fatal', 'offset_ratio',
                                    'offset/shortest-segment = %.1fx (layup %s off=%.3g m, min seg=%.3g m); PreVABS offset will collapse'
                                    % (ratio, max_off_layup, max_off, min_phys),
                                    True, 'uniform-resample the .dat to remove short segments (raises min segment) and/or add <normalize>1</normalize>'))
            elif ratio >= RATIO_WARN:
                issues.append(Issue('warn', 'offset_ratio',
                                    'offset/shortest-segment = %.1fx (fragile)' % ratio,
                                    True, 'resample .dat / add <normalize>1</normalize>'))

    # --- webs: point must be strictly inside the OML contour ----------------
    #   BOTH the web point (M = midpoint/chord in emit_prevabs) and the .dat are already
    #   normalized by chord, so compare in normalized space directly (do NOT divide by scale).
    if dat is not None:
        poly = dat[:, :2]
        for w in X['webs']:
            if len(w['xy']) >= 2 and not _point_in_poly(w['xy'][0], w['xy'][1], poly):
                issues.append(Issue('fatal', 'web_outside',
                                    'web point %s = (%.4f, %.4f) is outside the airfoil contour' % (w['name'], w['xy'][0], w['xy'][1]),
                                    False, 'recompute web attachment onto the contour (XML generation bug, not airfoil)'))

    # --- recommended layered-quad (to_solid_hex) params by airfoil type ------
    #   The near-circular THICK ROOT (s00/s01) must use the refined nr/nw/mesh_size
    #   or the structured loft inverts; prescribing it here lets the generator mesh
    #   FIRST TRY and skip the slow retry ladder. (info, not a defect.)
    quad_params = recommend_mesh_params(dat) if dat is not None else None
    if quad_params and quad_params['airfoil_kind'] == 'thick_circular_root':
        issues.append(Issue('info', 'thick_circular_root',
                            'near-circular thick root (t/c=%.3f): refine solid-mesh params -> nr=%d nw=%d mesh_size=%.3f'
                            % (quad_params['thickness_ratio'], quad_params['nr'],
                               quad_params['nw'], quad_params['mesh_size']),
                            True, 'to_solid_hex(mesh_size=%.3f, nr=%d, nw=%d)'
                            % (quad_params['mesh_size'], quad_params['nr'], quad_params['nw'])))

    # verdict
    sev = {'info': 0, 'warn': 1, 'fatal': 2}
    worst = max([sev[i['severity']] for i in issues], default=0)
    has_unfixable = any(i['severity'] == 'fatal' and not i['fixable'] for i in issues)
    has_fixable_fatal = any(i['severity'] == 'fatal' and i['fixable'] for i in issues)
    if has_unfixable:
        verdict = 'UNFIXABLE_USE_FALLBACK'
    elif has_fixable_fatal:
        verdict = 'FATAL_FIXABLE_IN_PREPROCESSING'
    elif worst == 1:
        verdict = 'WARN_LIKELY_OK'
    else:
        verdict = 'CLEAN'
    return dict(xml=xml_path, verdict=verdict, issues=issues,
                max_offset=max_off, scale=scale, quad_params=quad_params)


def _point_in_poly(x, y, poly):
    # vectorized ray-cast (numpy) -- ~0.05 ms vs ~1 ms for the Python loop
    xs = poly[:, 0]
    ys = poly[:, 1]
    xj = np.roll(xs, 1)
    yj = np.roll(ys, 1)
    cross = ((ys > y) != (yj > y)) & (x < (xj - xs) * (y - ys) / (yj - ys + 1e-30) + xs)
    return bool(int(np.count_nonzero(cross)) & 1)


# -------------------------------------------------------------------------- CLI
def _print_report(rep):
    print('\n=== %s ===' % os.path.basename(rep['xml']))
    order = {'fatal': 0, 'warn': 1, 'info': 2}
    for i in sorted(rep['issues'], key=lambda x: order[x['severity']]):
        tag = {'fatal': 'FATAL', 'warn': 'warn ', 'info': 'info '}[i['severity']]
        fx = 'fixable' if i['fixable'] else 'UNFIXABLE'
        print('  [%s] %-22s %s' % (tag, i['code'], i['message']))
        print('        (%s) fix: %s' % (fx, i['fix']))
    qp = rep.get('quad_params')
    if qp:
        print('  solid-mesh params [%s t/c=%.3f]: mesh_size=%.3f nr=%d nw=%d'
              % (qp['airfoil_kind'], qp['thickness_ratio'], qp['mesh_size'], qp['nr'], qp['nw']))
    print('  VERDICT: %s%s' % (rep['verdict'], '   (%.2f ms)' % rep['ms'] if 'ms' in rep else ''))


def main(argv=None):
    import time
    ap = argparse.ArgumentParser(description='Precondition-check a generated PreVABS cross-section XML.')
    ap.add_argument('paths', nargs='+', help='.xml file(s); globs allowed (quote them)')
    ap.add_argument('--json', action='store_true', help='machine-readable output')
    a = ap.parse_args(argv)
    files = []
    for p in a.paths:
        files.extend(sorted(glob.glob(p)) or [p])
    reports = []
    t_all = time.perf_counter()
    for f in files:
        if not os.path.exists(f):
            continue
        t0 = time.perf_counter()
        r = precheck(f)
        r['ms'] = (time.perf_counter() - t0) * 1000.0
        reports.append(r)
    total_ms = (time.perf_counter() - t_all) * 1000.0
    if a.json:
        import json
        print(json.dumps(reports, indent=2))
    else:
        for r in reports:
            _print_report(r)
        if len(reports) > 1:
            from collections import Counter
            c = Counter(r['verdict'] for r in reports)
            print('\nsummary: ' + ', '.join('%s=%d' % (k, v) for k, v in c.items()))
            print('timing: %d checks in %.1f ms  (%.2f ms/check avg)'
                  % (len(reports), total_ms, total_ms / max(len(reports), 1)))
    worst = 0
    for r in reports:
        if r['verdict'] == 'UNFIXABLE_USE_FALLBACK':
            worst = max(worst, 3)
        elif r['verdict'] == 'FATAL_FIXABLE_IN_PREPROCESSING':
            worst = max(worst, 2)
        elif r['verdict'] == 'WARN_LIKELY_OK':
            worst = max(worst, 1)
    return worst


if __name__ == '__main__':
    sys.exit(main())
