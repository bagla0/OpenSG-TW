'''Calibrate the s00/s01 (near-circular thick root) detector: thickness/chord per station,
and time the current precheck to confirm it is millisecond-fast.'''
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import precheck_prevabs as PC

XMLD = os.path.join(HERE, 'shell51', 'xml')
print('%-8s %8s %8s   %s' % ('tag', 't/c', 'ms', 'note'))
worked_default = {'s00', 's01'}   # these needed the fine-skin rung in gen_quad51
for i in range(51):
    tag = 'iea_s%02d' % i
    datp = os.path.join(XMLD, tag + '.dat')
    xmlp = os.path.join(XMLD, tag + '.xml')
    if not (os.path.exists(datp) and os.path.exists(xmlp)):
        continue
    _, dat = PC.parse_dat(datp)
    tc = float(dat[:, 1].max() - dat[:, 1].min())   # normalized thickness/chord
    t0 = time.perf_counter()
    n = 10
    for _ in range(n):
        PC.precheck(xmlp)
    ms = (time.perf_counter() - t0) / n * 1000.0
    note = 'ROOT (needed fine rung)' if ('s%02d' % i) in worked_default else ''
    print('%-8s %8.3f %8.2f   %s' % (tag, tc, ms, note))
