#!/usr/bin/env python
'''Produce s50.sg with the (now-launchable) v2.1.0-preview PreVABS layered mesher.
The layered offset calls gmsh for the TE bond areas; at mesh_size=0.04 gmsh cannot recover the TE
edge. Use the ORIGINAL geometry (no resample -> avoids the extra TE points that make the offset
DCEL a staircase) and try finer gmsh sizes so the constrained TE edge is recoverable. Short per-run
timeout so a stuck gmsh is skipped, not waited on. First success -> promote + convert to YAML.'''
import os
import sys
import shutil
import subprocess

HERE = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations'
PB = '/home/roger/a/bagla0/OpenSG_io/third_party/prevabs_bin/prevabs-v2.1.0-preview.20260508.3-linux-rhel9-x64/prevabs'
GCC12 = '/home/roger/a/bagla0/gcc12lib'          # newer libstdc++ so v2.1.0 launches
CVT = os.path.expanduser('~/OpenSG-TW-claude/third_party/OpenSG_io/scripts/convert_sg_to_yaml.py')
XMLDIR = os.path.join(HERE, 'shell51/xml')
WORK = os.path.join(HERE, 'shell51/xml/s50pn2')
SGV = os.path.join(HERE, 'shell51/sg_v201')
YAMLDIR = os.path.join(HERE, 'shell51/2d_hybrid')
NAME = 'iea_s50'

os.makedirs(WORK, exist_ok=True)
for f in (NAME + '.xml', NAME + '.dat', 'materials.xml'):
    src = os.path.join(XMLDIR, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(WORK, f))

env = dict(os.environ)
env['LD_LIBRARY_PATH'] = GCC12 + ':' + os.path.dirname(PB)


def set_mesh_size(ms):
    p = os.path.join(WORK, NAME + '.xml')
    txt = open(p).read()
    import re
    txt = re.sub(r'<mesh_size>[0-9.]+</mesh_size>', '<mesh_size>%g</mesh_size>' % ms, txt)
    open(p, 'w').write(txt)


def run(ms, timeout=100):
    set_mesh_size(ms)
    sg = os.path.join(WORK, NAME + '.sg')
    if os.path.exists(sg):
        os.remove(sg)
    try:
        r = subprocess.run([PB, '-i', NAME + '.xml', '--vabs', '--hm', '--gmsh-verbosity', '1'],
                           cwd=WORK, env=env, capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return sg, False, 'gmsh hung (> %ds), skipped' % timeout
    ok = os.path.exists(sg) and sum(1 for _ in open(sg)) > 100
    err = ''
    for L in out.splitlines():
        if any(k in L.lower() for k in ('recover', 'fatal', 'error')):
            err = L.strip()[:110]
    return sg, ok, err


print('=== v2.1.0 layered PreVABS on ORIGINAL s50 geometry, gmsh-size sweep ===')
result = None
for ms in [0.02, 0.015, 0.01, 0.008, 0.006, 0.03]:
    sg, ok, err = run(ms)
    print('  mesh_size=%-6g : %s' % (ms, ('SUCCESS (%d lines)' % sum(1 for _ in open(sg))) if ok else 'fail  [%s]' % err))
    if ok:
        result = sg
        break

if result:
    shutil.copy(result, os.path.join(SGV, NAME + '.sg'))
    if os.path.exists(result + '.mat'):
        shutil.copy(result + '.mat', os.path.join(SGV, NAME + '.sg.mat'))
    print('\npromoted -> %s' % os.path.join(SGV, NAME + '.sg'))
    outyaml = os.path.join(YAMLDIR, NAME + '_solid_prevabs.yaml')
    rc = subprocess.run([sys.executable, CVT, os.path.join(SGV, NAME + '.sg'), outyaml],
                        capture_output=True, text=True)
    print(rc.stdout.strip()[-300:])
    print('yaml -> %s' % outyaml)
else:
    print('\nNo gmsh size let the v2.1.0 layered mesher recover the s50 TE edge.')
    sys.exit(1)
