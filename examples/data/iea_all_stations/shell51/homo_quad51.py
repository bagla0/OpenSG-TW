'''homo_quad51.py -- homogenize the full IEA-22 blade from the pyNuMAD-quad 2-D cross-sections:
RM shell 6x6 for every 1d_yaml/*_shell.yaml   -> homo_rm/OpenSG_RM_<tag>.txt
JAX solid 6x6 for every pynumad_quad/*_solid.yaml -> homo_jax/OpenSG_JAX_<tag>.txt
Spawn-parallel (JAX forks deadlock). Writes log_rm.txt / log_jax.txt (parseable per-station time).'''
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/OpenSG-TW-claude')
XS = os.path.join(REPO, 'examples', 'TW-paper', 'xsec_paper')
RM_OUT = os.path.join(HERE, 'homo_rm'); os.makedirs(RM_OUT, exist_ok=True)
JX_OUT = os.path.join(HERE, 'homo_jax'); os.makedirs(JX_OUT, exist_ok=True)
QUAD = os.path.join(HERE, 'pynumad_quad')
Y1D = os.path.join(HERE, '1d_yaml')


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
        return (kind, tag, time.time() - t0, 'ok', float(C[0, 0]), float(C[3, 3]))
    except Exception as e:
        return (kind, tag, time.time() - t0, repr(e)[:120], 0.0, 0.0)


def main():
    tasks = [('rm', f) for f in sorted(glob.glob(os.path.join(Y1D, '*_shell.yaml')))]
    tasks += [('jax', f) for f in sorted(glob.glob(os.path.join(QUAD, '*_solid.yaml')))]
    print('homogenizing %d tasks (RM shells + JAX quad solids)' % len(tasks), flush=True)
    import multiprocessing as mp
    n = min(10, len(tasks))
    t0 = time.time()
    res = []
    with mp.get_context('spawn').Pool(n) as pool:
        for r in pool.imap_unordered(work, tasks):
            res.append(r)
            print('[%s] %-4s %-10s EA=%.4g GJ=%.4g [%.1fs] %s'
                  % (r[1], r[0], '', r[4], r[5], r[2], '' if r[3] == 'ok' else 'FAIL ' + r[3]), flush=True)
    wall = time.time() - t0
    rm = [r for r in res if r[0] == 'rm']
    jx = [r for r in res if r[0] == 'jax']
    with open(os.path.join(HERE, 'log_rm.txt'), 'w') as f:
        for r in sorted(rm, key=lambda x: x[1]):
            f.write('[%s] EA=%.4g GJ=%.4g [%.1fs]\n' % (r[1], r[4], r[5], r[2]))
    with open(os.path.join(HERE, 'log_jax.txt'), 'w') as f:
        for r in sorted(jx, key=lambda x: x[1]):
            f.write('[%s] EA=%.4g GJ=%.4g [%.1fs]\n' % (r[1], r[4], r[5], r[2]))
    okrm = sum(1 for r in rm if r[3] == 'ok'); okjx = sum(1 for r in jx if r[3] == 'ok')
    print('\nRM %d/%d, JAX %d/%d  homogenized' % (okrm, len(rm), okjx, len(jx)))
    print('HOMOGENIZATION wall time (RM shell + JAX solid, %d cores): %.1f s' % (n, wall))
    print('  sum of JAX-solid per-station compute: %.1f s' % sum(r[2] for r in jx))
    print('HOMO_QUAD_DONE', flush=True)


if __name__ == '__main__':
    main()
