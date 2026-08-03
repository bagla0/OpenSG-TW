import os, sys, numpy as np, yaml
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser('~/OpenSG-TW-claude/third_party/OpenSG_io'))
XS = os.path.expanduser('~/OpenSG-TW-claude/examples/TW-paper/xsec_paper')
REPO = os.path.abspath(os.path.join(XS, '..', '..', '..'))
for q in (XS, REPO, os.path.join(REPO, 'mitc_rm_segment')):
    sys.path.insert(0, q)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
from opensg_io.prevabs_xml import parse_prevabs_xml, to_shell
from opensg_io.converter import emit_opensg_yaml
from xsec_5v6_master import load_ring, ring_6dof

xml = os.path.join(HERE, 'xml', 'iea_r0247.xml')
tmp = '/tmp/r0247_test.yaml'

def ncount(p):
    return len(yaml.safe_load(open(p))['nodes'])

def ea_gj(p):
    C = np.asarray(ring_6dof(load_ring(p, center_ref=True)))
    return C[0, 0], C[3, 3]

# current (overwritten) 1d_yaml
cur = os.path.join(HERE, '1d_yaml', 'iea_r0247_shell.yaml')
print('current 1d_yaml r0247 : %d nodes' % ncount(cur))

# fresh via to_shell (the ORIGINAL shipped path, no shift, default fraction)
to_shell(xml, tmp)
print('to_shell default      : %d nodes  EA=%.4e GJ=%.4e' % (ncount(tmp), *ea_gj(tmp)))

# parse + emit fraction=0.5 (no shift)
cs = parse_prevabs_xml(xml)
emit_opensg_yaml(cs, tmp, fraction=0.5)
print('parse+emit frac=0.5   : %d nodes  EA=%.4e GJ=%.4e' % (ncount(tmp), *ea_gj(tmp)))

# parse + emit fraction=0.0 (OML, no shift)
cs = parse_prevabs_xml(xml)
emit_opensg_yaml(cs, tmp, fraction=0.0)
print('parse+emit frac=0.0   : %d nodes  EA=%.4e GJ=%.4e' % (ncount(tmp), *ea_gj(tmp)))
