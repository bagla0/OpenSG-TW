'''confirm the .sg orientation (theta1,theta3) inverse reproduces the yaml element frame exactly.'''
import os, numpy as np, yaml
FB = '/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations/shell51/fallback_yaml'


def _row(x):
    if isinstance(x, list):
        return [float(v) for v in (x[0].split() if isinstance(x[0], str) else x)]
    return [float(v) for v in str(x).split()]


def frame(t1, t3):
    c1, s1 = np.cos(np.radians(t1)), np.sin(np.radians(t1))
    c3, s3 = np.cos(np.radians(t3)), np.sin(np.radians(t3))
    return np.array([s3 * c1, s3 * s1, c3, c3 * c1, c3 * s1, -s3, -s1, c1, 0.0])


def inv(o):
    o = np.array(o); e1 = o[0:3]; e3 = o[6:9]
    t1 = np.degrees(np.arctan2(-e3[0], e3[1]))
    c1, s1 = np.cos(np.radians(t1)), np.sin(np.radians(t1))
    t3 = np.degrees(np.arctan2(e1[0] * c1 + e1[1] * s1, e1[2]))
    return t1, t3


for st in ('s02', 's50'):
    d = yaml.safe_load(open(os.path.join(FB, 'iea_%s_solid.yaml' % st)))
    th3s = []
    err = 0.0
    for o in d['elementOrientations']:
        o = _row(o); t1, t3 = inv(o); th3s.append(t3)
        err = max(err, float(np.max(np.abs(frame(t1, t3) - np.array(o)))))
    print('%s: orient round-trip max err = %.2e   theta3 range [%.1f, %.1f] deg  (%d elems)'
          % (st, err, min(th3s), max(th3s), len(th3s)))
