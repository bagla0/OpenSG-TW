#!/usr/bin/env python
'''
condition_prevabs.py  --  shape-PRESERVING preprocessing of a PreVABS cross-section XML/.dat.

Given an XML that precheck_prevabs.py flagged as FATAL_FIXABLE, apply input-side fixes that
make PreVABS's inward offset succeed WITHOUT changing the true airfoil shape, re-check, and
report whether it is now clean.  If a fatal issue remains that no preprocessing can resolve
(e.g. the laminate is thicker than the local section can hold), it stays fatal and the caller
routes to the pyNuMAD-style fallback.

Fixes (each preserves the airfoil to a stated tolerance):
  1. UNIFORM ARC-LENGTH RESAMPLE of the .dat  -> removes pathologically short (sub-mm)
     baseline segments that make the inward offset collapse.  Linear resample along the SAME
     polyline, so max chordal deviation < original local sagitta (<< mesh_size); LE (min-x)
     and both TE endpoints are kept as exact samples so no shape feature is lost.
  2. OPEN A MINIMUM TRAILING-EDGE GAP  -> if the TE is a near-zero-thickness sliver, move ONLY
     the two TE endpoints apart normal to the chord by the deficit; interior is untouched.
  3. DECLUMP DIVIDING POINTS  -> enforce a minimum x-spacing between consecutive same-side
     dividing points (fixes the LE_XMIN clamp collision that makes d6==d7 -> zero-length
     baseline).  Only moves a layup boundary a fraction of a percent chord.
  4. ADD <normalize>1</normalize>  -> PreVABS re-parametrises the baseline by arc length before
     offsetting, its own recommended remedy for fragile offsets.

USAGE
    python condition_prevabs.py path/to/iea_s02.xml               # writes <dir>/cond/iea_s02.xml (+ .dat, materials.xml)
    python condition_prevabs.py iea_s02.xml --out somedir
    python condition_prevabs.py iea_s02.xml --dry-run             # report planned fixes only

Importable:
    from condition_prevabs import condition
    info = condition('iea_s02.xml', out_dir='cond')   # -> dict(new_xml, fixes, verdict_before, verdict_after)
'''
import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

import precheck_prevabs as P

MIN_DIV_GAP = 0.01      # normalized-x minimum spacing between same-side dividing points
TE_TARGET_GAP = 1e-3    # m; open TE to at least this (physical)
RESAMPLE_MIN_SEG = 1.5e-3   # m; target: no baseline segment shorter than this after resample


# --------------------------------------------------------------- .dat helpers
def resample_dat(pts, scale, target_min_seg=RESAMPLE_MIN_SEG, keep_le=True):
    '''Uniform arc-length resample of a closed airfoil polyline to remove short segments.
    Chooses N so the mean physical spacing ~ target_min_seg, but never fewer than the input.
    Keeps the first/last (TE) points and inserts the LE (min-x) as an exact sample.'''
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    phys_total = total * scale
    # REDISTRIBUTE the existing points uniformly to kill sub-mm clusters; never OVER-refine
    # (too many points make PreVABS's layered-offset DCEL produce an unsplittable "staircase").
    N = max(len(pts), min(2 * len(pts), int(np.ceil(phys_total / target_min_seg)) + 1))
    su = np.linspace(0.0, total, N)
    if keep_le:                                   # force a sample exactly at the LE (min x)
        le = int(np.argmin(pts[:, 0]))
        su = np.unique(np.concatenate([su, [s[le]]]))
    x = np.interp(su, s, pts[:, 0])
    y = np.interp(su, s, pts[:, 1])
    out = np.column_stack([x, y])
    out[0] = pts[0]                               # exact TE endpoints
    out[-1] = pts[-1]
    return out


def open_te(pts, scale, target=TE_TARGET_GAP):
    '''If the TE gap (first vs last point) is below target (physical), push ONLY the two TE
    endpoints apart, normal to the local chord direction. Interior untouched.'''
    gap = np.linalg.norm(pts[0] - pts[-1]) * scale
    if gap >= target:
        return pts, False
    need = (target - gap) / scale / 2.0           # normalized half-deficit
    # local chord tangent at TE ~ direction from TE to a slightly-interior average point
    mid = 0.5 * (pts[0] + pts[-1])
    interior = 0.5 * (pts[3] + pts[-4]) if len(pts) > 8 else 0.5 * (pts[1] + pts[-2])
    t = mid - interior
    n = np.linalg.norm(t)
    if n < 1e-12:
        nrm = np.array([0.0, 1.0])
    else:
        t = t / n
        nrm = np.array([-t[1], t[0]])             # normal to chord at TE
    pts = pts.copy()
    pts[0] = pts[0] + need * nrm
    pts[-1] = pts[-1] - need * nrm
    return pts, True


def declump_points(points, min_gap=MIN_DIV_GAP):
    '''Enforce a minimum x-gap between consecutive same-side dividing points. Returns
    {name: new_x} for the points that moved.'''
    moved = {}
    for side in ('top', 'bottom'):
        pts = sorted(((nm, p['x']) for nm, p in points.items() if p.get('side') == side),
                     key=lambda kv: kv[1])
        for i in range(1, len(pts)):
            na, xa = pts[i - 1]
            nb, xb = pts[i]
            if xb - xa < min_gap:
                newx = min(xa + min_gap, 1.0 - 1e-4)
                pts[i] = (nb, newx)
                moved[nb] = newx
    return moved


# ----------------------------------------------------------------- conditioner
def condition(xml_path, out_dir=None, dry_run=False, verbose=True):
    d = os.path.dirname(os.path.abspath(xml_path))
    base = os.path.basename(xml_path)
    out_dir = out_dir or os.path.join(d, 'cond')
    rep_before = P.precheck(xml_path)

    X = P.parse_xml(xml_path)
    scale = X['scale'] or 1.0
    fixes = []

    # load .dat
    dat_path = os.path.join(d, X['dat_ref']) if X['dat_ref'] else None
    _, dat = P.parse_dat(dat_path) if dat_path and os.path.exists(dat_path) else ('', None)

    codes = {i['code'] for i in rep_before['issues']}

    new_dat = dat
    if dat is not None:
        # 1. resample if short segments / offset ratio flagged
        if codes & {'dat_short_segments', 'offset_ratio', 'dat_coincident_pts'}:
            n0 = len(new_dat)
            new_dat = resample_dat(new_dat, scale)
            fixes.append('resampled .dat %d->%d pts (uniform arc-length; airfoil preserved)' % (n0, len(new_dat)))
        # 2. open TE
        if 'te_sliver' in codes:
            new_dat, did = open_te(new_dat, scale)
            if did:
                fixes.append('opened trailing edge to >= %.1g m (2 endpoints only)' % TE_TARGET_GAP)

    # 3. declump dividing points
    moved = declump_points(X['points'])
    if moved:
        fixes.append('declumped %d dividing point(s) to >= %.3g x-gap: %s'
                     % (len(moved), MIN_DIV_GAP, ', '.join('%s->%.4f' % (k, v) for k, v in moved.items())))

    # 4. always add <normalize>1</normalize> when we had a fatal-fixable offset problem
    add_normalize = bool(codes & {'offset_ratio', 'dat_short_segments'})
    if add_normalize:
        fixes.append('added <normalize>1</normalize> to <general>')

    if dry_run:
        if verbose:
            print('[%s] planned fixes:' % base)
            for f in fixes:
                print('   -', f)
            print('   verdict before:', rep_before['verdict'])
        return dict(new_xml=None, fixes=fixes, verdict_before=rep_before['verdict'], verdict_after=None, dry_run=True)

    os.makedirs(out_dir, exist_ok=True)
    # write conditioned .dat (same basename so the XML ref is unchanged)
    if new_dat is not None and X['dat_ref']:
        with open(os.path.join(out_dir, X['dat_ref']), 'w') as f:
            f.write((X['dat_ref'].rsplit('.', 1)[0]) + '\n')
            for xx, yy in new_dat:
                f.write('% .8f % .8f\n' % (xx, yy))
    # copy materials.xml
    mx = os.path.join(d, 'materials.xml')
    if os.path.exists(mx):
        shutil.copy(mx, os.path.join(out_dir, 'materials.xml'))

    # edit + write the XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    g = root.find('general')
    if add_normalize and g is not None and g.find('normalize') is None:
        ET.SubElement(g, 'normalize').text = '1'
    if moved:
        for e in root.iter('point'):
            if e.get('name') in moved:
                e.text = '%.6f' % moved[e.get('name')]
    tree.write(os.path.join(out_dir, base), encoding='unicode')

    rep_after = P.precheck(os.path.join(out_dir, base))
    if verbose:
        print('[%s]' % base)
        for f in fixes:
            print('   fix:', f)
        print('   verdict: %s  ->  %s' % (rep_before['verdict'], rep_after['verdict']))
        if rep_after['verdict'] in ('UNFIXABLE_USE_FALLBACK', 'FATAL_FIXABLE_IN_PREPROCESSING'):
            remain = [i for i in rep_after['issues'] if i['severity'] == 'fatal']
            for i in remain:
                print('   STILL FATAL: %s (%s)' % (i['code'], 'fixable' if i['fixable'] else 'UNFIXABLE'))
        print('   wrote:', os.path.join(out_dir, base))
    return dict(new_xml=os.path.join(out_dir, base), fixes=fixes,
                verdict_before=rep_before['verdict'], verdict_after=rep_after['verdict'])


def main(argv=None):
    ap = argparse.ArgumentParser(description='Shape-preserving preprocessing of a PreVABS XML.')
    ap.add_argument('xml')
    ap.add_argument('--out', default=None, help='output dir (default <xmldir>/cond)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    r = condition(a.xml, out_dir=a.out, dry_run=a.dry_run)
    return 0 if r.get('verdict_after') in (None, 'CLEAN', 'WARN_LIKELY_OK') else 2


if __name__ == '__main__':
    sys.exit(main())
