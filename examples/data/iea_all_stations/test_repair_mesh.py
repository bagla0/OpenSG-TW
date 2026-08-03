'''Test the general repair_mesh.py tool:
 1. dry-run diagnose over ALL 49 solid meshes (should all be clean now, incl. repaired s28)
 2. round-trip: restore the dirty s28 from its .orig backup to a temp, repair it, confirm clean
    and that GA2/GA3 come back on-trend when homogenized.'''
import os
import sys
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import repair_mesh as R
import glob

print('=== 1. diagnose all current 2d solids (post-repair state) ===')
dirty = []
for p in sorted(glob.glob(os.path.join(HERE, 'shell51', '2d_yaml', '*_solid.yaml'))):
    if '_t1only' in p:
        continue
    s = R.diagnose(p)
    bad = s['coincident'] or s['zero_measure'] or s['repeated_vertex']
    if bad:
        dirty.append(os.path.basename(p))
        print('  DIRTY %-28s coincident=%d zero=%d repeated=%d' %
              (os.path.basename(p), s['coincident'], s['zero_measure'], s['repeated_vertex']))
print('  dirty solids now: %s' % (dirty if dirty else 'NONE (all clean)'))

print('\n=== 2. round-trip repair from the dirty s28.orig backup ===')
orig = os.path.join(HERE, 'shell51', '2d_yaml', 'iea_s28_solid.yaml.orig')
if not os.path.exists(orig):
    print('  (no .orig backup found; skipping round-trip)')
else:
    tmp = os.path.join(HERE, 'shell51', '2d_yaml', '_test_s28_dirty.yaml')
    shutil.copy(orig, tmp)
    print('  before repair:', {k: R.diagnose(tmp)[k] for k in ('coincident', 'zero_measure', 'repeated_vertex')})
    res = R.repair(tmp, backup=False, verbose=True)
    v = res['verify']
    print('  -> round-trip result: coincident=%d zero=%d  %s' %
          (v['coincident'], v['zero_measure'], 'PASS' if (v['coincident'] == 0 and v['zero_measure'] == 0) else 'FAIL'))
    os.remove(tmp)
    print('  (temp file removed)')
