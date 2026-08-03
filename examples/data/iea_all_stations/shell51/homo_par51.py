'''homo_par51.py -- SPAWN-parallel homogenization of the shell51 stations (fork deadlocks JAX):
RM shell 6x6 for every 1d_yaml/*_shell.yaml  -> homo_rm/OpenSG_RM_<tag>.txt
JAX solid 6x6 for every 2d_yaml/*_solid.yaml -> homo_jax/OpenSG_JAX_<tag>.txt
(meshes must already be shifted to x1 by refaxis_shift51.py). Center_ref=True mid-surface for RM.'''
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
REPO = os.path.expanduser('~/OpenSG-TW-claude')
XS = os.path.join(REPO, 'examples', 'TW-paper', 'xsec_paper')
RM_OUT = os.path.join(HERE, 'homo_rm'); os.makedirs(RM_OUT, exist_ok=True)
JX_OUT = os.path.join(HERE, 'homo_jax'); os.makedirs(JX_OUT, exist_ok=True)


def work(task):
    kind, path = task
    tag = os.path.basename(path).replace('_shell.yaml', '').replace('_solid.yaml', '')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    for q in (XS, REPO, os.path.join(REPO, 'opensg_jax'), os.path.join(REPO, 'mitc_rm_segment')):
        if q not in sys.path:
            sys.path.insert(0, q)
    import jax
    jax.config.update('jax_enable_x64', True)
    t0 = time.time()
    try:
        if kind == 'rm':
            from xsec_5v6_master import load_ring, ring_6dof
            C = np.asarray(ring_6dof(load_ring(path, center_ref=True)))
            np.savetxt(os.path.join(RM_OUT, 'OpenSG_RM_%s.txt' % tag), C)
        else:
            from opensg_jax.fe_jax.solid_timo import compute_timo_from_yaml
            C = np.asarray(compute_timo_from_yaml(path, verbose=False))
            np.savetxt(os.path.join(JX_OUT, 'OpenSG_JAX_%s.txt' % tag), C)
        return '[%s %-10s] EA=%.4g GJ=%.4g  [%.1fs]' % (kind, tag, C[0, 0], C[3, 3], time.time() - t0)
    except Exception as e:
        return '[%s %-10s] FAIL %s' % (kind, tag, repr(e)[:150])


def main():
    tasks = [('rm', f) for f in sorted(glob.glob(os.path.join(HERE, '1d_yaml', '*_shell.yaml')))]
    tasks += [('jax', f) for f in sorted(glob.glob(os.path.join(HERE, '2d_yaml', '*_solid.yaml')))
              if 't1only' not in os.path.basename(f)]
    print('homogenizing %d tasks (RM shells + JAX solids), spawn pool' % len(tasks), flush=True)
    import multiprocessing as mp
    n = min(10, len(tasks))
    with mp.get_context('spawn').Pool(n) as pool:
        for r in pool.imap_unordered(work, tasks):
            print(r, flush=True)
    print('HOMO_PAR_DONE', flush=True)


if __name__ == '__main__':
    main()
