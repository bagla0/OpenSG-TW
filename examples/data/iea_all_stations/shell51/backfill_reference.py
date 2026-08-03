'''backfill_reference.py -- the 51 existing 1-D shell yamls were built center-ref (fraction=0.5) but
predate the `reference` field.  Append `reference: center` to any that lack it, so
dehom_rm.build_rm_bundle reads the reference from the yaml (single source of truth).'''
import os, glob
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '1d_yaml')
n = 0
for f in sorted(glob.glob(os.path.join(D, 'iea_s*_shell.yaml'))):
    txt = open(f).read()
    if any(l.strip().startswith('reference:') for l in txt.splitlines()):
        continue
    if not txt.endswith('\n'):
        txt += '\n'
    open(f, 'w').write(txt + 'reference: center\n')
    n += 1
print('backfilled reference: center into %d yaml(s) (%d already had it)'
      % (n, len(glob.glob(os.path.join(D, 'iea_s*_shell.yaml'))) - n))
