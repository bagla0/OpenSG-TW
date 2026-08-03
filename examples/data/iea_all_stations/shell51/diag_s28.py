'''Diagnose the s28 transverse-shear (GA2/GA3) halving.
Compare s28 to neighbours s26,s27,s29,s30; inspect the 1d shell mesh (segments, webs, layup, element lengths).'''
import os
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
LBL = ['EA', 'GA2', 'GA3', 'GJ', 'EI2', 'EI3']


def d6(p):
    M = np.loadtxt(p)
    return [M[k, k] for k in range(6)]


print('=== shell diagonal, s26..s30 (neighbours of s28) ===')
print('%-5s %12s %12s %12s %12s %12s' % ('', 's26', 's27', 's28', 's29', 's30'))
cols = {}
for tag in ('s26', 's27', 's28', 's29', 's30'):
    p = os.path.join(HERE, 'homo_rm', 'OpenSG_RM_iea_%s.txt' % tag)
    cols[tag] = d6(p) if os.path.exists(p) else [np.nan] * 6
for k in range(6):
    print('%-5s %12.4e %12.4e %12.4e %12.4e %12.4e' % (
        LBL[k], cols['s26'][k], cols['s27'][k], cols['s28'][k], cols['s29'][k], cols['s30'][k]))

print('\n=== solid diagonal (JAX) s26..s30 ===')
scols = {}
for tag in ('s26', 's27', 's28', 's29', 's30'):
    p = os.path.join(HERE, 'homo_jax', 'OpenSG_JAX_iea_%s.txt' % tag)
    scols[tag] = d6(p) if os.path.exists(p) else [np.nan] * 6
for k in range(6):
    print('%-5s %12.4e %12.4e %12.4e %12.4e %12.4e' % (
        LBL[k], scols['s26'][k], scols['s27'][k], scols['s28'][k], scols['s29'][k], scols['s30'][k]))

print('\n=== shell/solid GA ratio ===')
for tag in ('s26', 's27', 's28', 's29', 's30'):
    print('%s GA2 %.3f  GA3 %.3f' % (tag, cols[tag][1] / scols[tag][1], cols[tag][2] / scols[tag][2]))


def mesh_info(tag):
    p = os.path.join(HERE, '1d_yaml', 'iea_%s_shell.yaml' % tag)
    d = yaml.safe_load(open(p))
    nn = len(d['nodes'])
    # elements / connectivity
    ne = 0
    for key in ('elements', 'element_connectivity', 'connectivity'):
        if key in d:
            ne = len(d[key])
            break
    # subdomains / sets keys
    keys = list(d.keys())
    # layup/material sets
    setinfo = {}
    for key in d:
        if 'set' in key.lower() or 'subdomain' in key.lower() or 'layup' in key.lower() or 'segment' in key.lower():
            v = d[key]
            setinfo[key] = len(v) if hasattr(v, '__len__') else v
    return nn, ne, keys, setinfo


print('\n=== 1d shell mesh info ===')
for tag in ('s26', 's27', 's28', 's29', 's30'):
    nn, ne, keys, setinfo = mesh_info(tag)
    print('%s nodes=%d elems=%d' % (tag, nn, ne))
    print('   keys=%s' % keys)
    print('   sets=%s' % setinfo)
