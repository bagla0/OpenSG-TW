#!/usr/bin/env python
'''
repair_mesh.py  --  general OpenSG cross-section mesh repair.

Fixes the "degenerate mesh" defects that a mesher (PreVABS / gmsh) can leave in an
OpenSG YAML and that silently corrupt a homogenization (most often the delicate
transverse-shear terms GA2/GA3 of the 2-D solid, or a solve that returns NaN):

  1. COINCIDENT NODES   two (or more) nodes at the same coordinate that the mesher
                        failed to weld -> a topological "crack" and zero-area elements.
  2. ZERO-AREA / ZERO-LENGTH ELEMENTS   a triangle/quad with collapsed vertices
                        (singular element Jacobian -> pollutes the global solve).
  3. (optional) SLIVER ELEMENTS   near-degenerate elements (quality << 1).

The repair is a standard mesh "weld":
    * merge coincident nodes to a single canonical node,
    * drop the elements that collapse (repeated vertex) or have ~zero measure,
    * renumber the surviving nodes,
    * remap element labels inside `sets` and keep every element-parallel array
      (`elementOrientations`, and `sections` when it is per-element) in lock-step,
    * write the YAML back in its original textual format.

Because the dropped elements had ~zero measure, welding leaves NO hole: the mesh
stays conformal and the stiffness is unchanged except for removing the numerical
poison.  Works on 2-D solid meshes (tri/quad) and 1-D shell meshes (line elements).

USAGE
    python repair_mesh.py path/to/mesh.yaml                 # repair in place (+ .orig backup)
    python repair_mesh.py mesh.yaml --out fixed.yaml        # write elsewhere
    python repair_mesh.py mesh.yaml --dry-run               # only diagnose, change nothing
    python repair_mesh.py mesh.yaml --drop-slivers --sliver-q 0.02
    python repair_mesh.py 'dir/*_solid.yaml'                # glob many (quote it)

Importable:
    from repair_mesh import diagnose, repair
    stats = diagnose('mesh.yaml')          # dict of defect counts, no writing
    stats = repair('mesh.yaml', dry_run=False)
'''
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import yaml


# --------------------------------------------------------------------------- IO
def _toks(row):
    '''Tokenize a nodes/elements row for either YAML style:
       "- [1.0 2.0 0.0]"  (parsed as a 1-element string list)  OR
       "- [1.0, 2.0, 0.0]" (parsed as a real list).'''
    if isinstance(row, str):
        return row.split()
    if isinstance(row, (list, tuple)) and len(row) == 1 and isinstance(row[0], str):
        return row[0].split()
    return [str(x) for x in row]


def _detect_style(path, key):
    '''Return 'comma' or 'space' by looking at the first data line under `key`.'''
    with open(path) as f:
        inblk = False
        for ln in f:
            if ln.rstrip().startswith(key + ':'):
                inblk = True
                continue
            if inblk:
                s = ln.strip()
                if s.startswith('- ['):
                    return 'comma' if ',' in s else 'space'
                if s and not s.startswith('-'):
                    return 'space'
    return 'space'


def load(path):
    d = yaml.safe_load(open(path))
    nodes = np.array([[float(x) for x in _toks(r)] for r in d['nodes']], dtype=float)
    elems = [[int(x) for x in _toks(r)] for r in d['elements']]
    return d, nodes, elems


# ---------------------------------------------------------------- geometry util
def _measure(pts):
    '''Signed measure of an element from its corner points:
       2 pts -> length ; >=3 pts -> polygon area (shoelace on first-ring corners).'''
    if len(pts) == 2:
        return np.linalg.norm(pts[1] - pts[0])
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _quality(pts):
    '''Shape quality in (0,1]; ~1 good, ->0 sliver.  Area elements only.'''
    if len(pts) < 3:
        return 1.0
    A = _measure(pts)
    per = sum(np.linalg.norm(pts[(i + 1) % len(pts)] - pts[i]) for i in range(len(pts)))
    if per <= 0:
        return 0.0
    # normalized so an equilateral triangle / square -> 1.0
    norm = (4.0 * np.sqrt(3.0)) if len(pts) == 3 else 16.0
    return norm * A / (per * per)


# --------------------------------------------------------------------- diagnose
def diagnose(path, round_dec=6, area_tol=1e-10, sliver_q=0.02):
    d, nodes, elems = load(path)
    xy = nodes[:, :2]
    # coincident nodes
    seen, coinc = {}, 0
    for i, nd in enumerate(xy):
        k = tuple(np.round(nd, round_dec))
        if k in seen:
            coinc += 1
        else:
            seen[k] = i
    zero = repeat = sliver = 0
    minA = np.inf
    for e in elems:
        idx = [i - 1 for i in e]                 # yaml is 1-indexed
        if len(set(idx)) < len(idx):
            repeat += 1
        pts = xy[idx]
        m = _measure(pts)
        minA = min(minA, m)
        if m < area_tol:
            zero += 1
        elif _quality(pts) < sliver_q:
            sliver += 1
    return dict(path=path, nn=len(nodes), ne=len(elems), arity=len(elems[0]) if elems else 0,
                coincident=coinc, zero_measure=zero, repeated_vertex=repeat,
                slivers=sliver, min_measure=(0.0 if not np.isfinite(minA) else minA))


# ----------------------------------------------------------------------- repair
def _fmt_row(vals, style, intfmt=False):
    if intfmt:
        body = (', ' if style == 'comma' else ' ').join(str(int(v)) for v in vals)
    else:
        body = (', ' if style == 'comma' else ' ').join('%.8f' % v for v in vals)
    return '- [%s]' % body


def repair(path, out=None, round_dec=6, area_tol=1e-10, drop_slivers=False,
           sliver_q=0.02, backup=True, dry_run=False, verbose=True):
    d, nodes, elems = load(path)
    orient = d.get('elementOrientations')
    sets = d.get('sets')
    sections = d.get('sections')
    materials = d.get('materials')
    nn0, ne0 = len(nodes), len(elems)
    ndim = nodes.shape[1]
    xy = nodes[:, :2]

    before = diagnose(path, round_dec, area_tol, sliver_q)

    # 1. weld coincident nodes  (old 0-idx -> canonical 0-idx)
    canon, node_map = {}, np.arange(nn0)
    for i in range(nn0):
        k = tuple(np.round(xy[i], round_dec))
        if k in canon:
            node_map[i] = canon[k]
        else:
            canon[k] = i
    n_welded = int((node_map != np.arange(nn0)).sum())

    # 2. mark elements to drop after welding
    keep = np.ones(ne0, dtype=bool)
    for j, e in enumerate(elems):
        w = [node_map[i - 1] for i in e]         # welded, 0-idx
        if len(set(w)) < len(w):
            keep[j] = False                       # collapsed
            continue
        m = _measure(xy[w])
        if m < area_tol:
            keep[j] = False
        elif drop_slivers and _quality(xy[w]) < sliver_q:
            keep[j] = False
    n_dropped = int((~keep).sum())

    # 3. surviving nodes (only those referenced by a kept element) -> new 1-idx
    used = sorted({node_map[i - 1] for e, k in zip(elems, keep) if k for i in e})
    old2new = {old: j + 1 for j, old in enumerate(used)}
    new_nodes = nodes[used]

    # 4. rebuild elements + element-parallel arrays + old->new label map
    new_elems, new_orient, new_sections = [], [], []
    label_map, new_lab = {}, 0
    sections_is_parallel = isinstance(sections, list) and len(sections) == ne0
    for old_lab, (e, k) in enumerate(zip(elems, keep), start=1):
        if not k:
            continue
        new_lab += 1
        label_map[old_lab] = new_lab
        new_elems.append([old2new[node_map[i - 1]] for i in e])
        if orient is not None:
            new_orient.append(orient[old_lab - 1])
        if sections_is_parallel:
            new_sections.append(sections[old_lab - 1])

    # 5. remap element labels inside sets
    new_sets = None
    if sets is not None:
        new_sets = {}
        for cat, groups in sets.items():          # usually {'element': [...]}
            new_groups = []
            for g in groups:
                if isinstance(g, dict) and 'labels' in g:
                    lbl = [label_map[l] for l in g['labels'] if l in label_map]
                    ng = dict(g)
                    ng['labels'] = lbl
                    new_groups.append(ng)
                else:
                    new_groups.append(g)
            new_sets[cat] = new_groups

    after = dict(nn=len(new_nodes), ne=len(new_elems),
                 welded=n_welded, dropped=n_dropped)
    if verbose:
        print('[%s]' % os.path.basename(path))
        print('  before: nn=%d ne=%d  coincident=%d zero-measure=%d repeated=%d slivers=%d min-measure=%.3e'
              % (before['nn'], before['ne'], before['coincident'], before['zero_measure'],
                 before['repeated_vertex'], before['slivers'], before['min_measure']))
        print('  action: welded %d node(s), dropped %d element(s)%s'
              % (n_welded, n_dropped, ' (+slivers)' if drop_slivers else ''))
        print('  after : nn=%d ne=%d' % (len(new_nodes), len(new_elems)))

    if before['coincident'] == 0 and before['zero_measure'] == 0 and before['repeated_vertex'] == 0 \
            and not (drop_slivers and before['slivers']):
        if verbose:
            print('  -> mesh already clean, nothing to do.')
        return dict(before=before, after=after, changed=False)

    if dry_run:
        if verbose:
            print('  -> dry-run: no file written.')
        return dict(before=before, after=after, changed=False, dry_run=True)

    # 6. write YAML preserving the original textual style, key order intact
    nstyle = _detect_style(path, 'nodes')
    estyle = _detect_style(path, 'elements')
    lines = []
    for key in d.keys():
        if key == 'nodes':
            lines.append('nodes:')
            for nd in new_nodes:
                lines.append(_fmt_row(nd[:ndim], nstyle))
        elif key == 'elements':
            lines.append('elements:')
            for e in new_elems:
                lines.append(_fmt_row(e, estyle, intfmt=True))
        elif key == 'sets':
            lines.append('sets:')
            for cat, groups in new_sets.items():
                lines.append('  %s:' % cat)
                for g in groups:
                    if isinstance(g, dict) and 'labels' in g:
                        lines.append('  - name: %s' % g.get('name', ''))
                        lines.append('    labels: [%s]' % ', '.join(str(x) for x in g['labels']))
                    else:
                        lines.append('  - ' + yaml.safe_dump(g, default_flow_style=True).strip())
        elif key == 'elementOrientations':
            lines.append('elementOrientations:')
            for o in new_orient:
                lines.append('- [%s]' % ', '.join(repr(float(x)) for x in o))
        elif key == 'sections' and sections_is_parallel:
            lines.append('sections:')
            for s in new_sections:
                lines.append('- ' + yaml.safe_dump(s, default_flow_style=True).strip())
        else:
            # untouched block (materials, non-parallel sections, misc): dump verbatim
            lines.append(yaml.safe_dump({key: d[key]}, default_flow_style=False,
                                        sort_keys=False).rstrip('\n'))

    text = '\n'.join(lines) + '\n'
    dest = out or path
    if backup and dest == path:
        shutil.copy(path, path + '.orig')
    open(dest, 'w').write(text)

    # 7. verify the result re-parses and is clean
    chk = diagnose(dest, round_dec, area_tol, sliver_q)
    if orient is not None:
        d2 = yaml.safe_load(open(dest))
        assert len(d2['elementOrientations']) == chk['ne'], 'orientation/element length mismatch'
    if verbose:
        print('  verify: reparsed nn=%d ne=%d  coincident=%d zero-measure=%d  -> %s'
              % (chk['nn'], chk['ne'], chk['coincident'], chk['zero_measure'],
                 'CLEAN' if (chk['coincident'] == 0 and chk['zero_measure'] == 0) else 'STILL DIRTY'))
        print('  wrote: %s%s' % (dest, '  (backup %s.orig)' % path if backup and dest == path else ''))
    return dict(before=before, after=after, verify=chk, changed=True)


# -------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description='Repair degenerate OpenSG cross-section meshes (weld + drop).')
    ap.add_argument('paths', nargs='+', help='YAML mesh file(s); globs allowed (quote them)')
    ap.add_argument('--out', default=None, help='output path (single input only); default in-place')
    ap.add_argument('--round', type=int, default=6, dest='round_dec',
                    help='coordinate decimals for coincidence test (default 6 = 1e-6 m)')
    ap.add_argument('--area-tol', type=float, default=1e-10, help='drop elements with measure below this')
    ap.add_argument('--drop-slivers', action='store_true', help='also drop near-degenerate slivers')
    ap.add_argument('--sliver-q', type=float, default=0.02, help='sliver quality threshold (0..1)')
    ap.add_argument('--dry-run', action='store_true', help='diagnose only, write nothing')
    ap.add_argument('--no-backup', action='store_true', help='do not write a .orig backup')
    a = ap.parse_args(argv)

    files = []
    for p in a.paths:
        files.extend(sorted(glob.glob(p)) or [p])
    if a.out and len(files) != 1:
        ap.error('--out requires exactly one input file')

    n_changed = n_dirty = 0
    for f in files:
        if not os.path.exists(f):
            print('[%s] NOT FOUND' % f)
            continue
        r = repair(f, out=a.out, round_dec=a.round_dec, area_tol=a.area_tol,
                   drop_slivers=a.drop_slivers, sliver_q=a.sliver_q,
                   backup=not a.no_backup, dry_run=a.dry_run)
        if r.get('changed'):
            n_changed += 1
        v = r.get('verify')
        if v and (v['coincident'] or v['zero_measure']):
            n_dirty += 1
    if len(files) > 1:
        print('\nsummary: %d/%d file(s) repaired%s' %
              (n_changed, len(files), (', %d STILL DIRTY' % n_dirty) if n_dirty else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
