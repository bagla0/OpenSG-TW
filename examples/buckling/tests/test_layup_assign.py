"""test_layup_assign.py -- does arc-length-conserving (sub-sampled) layup assignment fix the composite
arc-length budget?

Truth: the station's OWN ring (native yaml layup map) integrated with the reduced axial identity
    EA = oint (A11 - A12^2/A22) ds        (thin wall, free hoop -- the CORRECT identity; raw oint A11 ds
                                           equals EA/(1-nu^2) only for isotropic walls)
The conformal section must reproduce that same integral.  Nearest-midpoint assignment does not (the stiff
spar cap is over-assigned arc length); sub-sampling should.

Expected: ratio(conformal/ring) 1.03-1.33 (nsub=1, i.e. the old lookup) -> ~1.00 (nsub=12).
"""
import os, sys
import numpy as np
import yaml
BUCK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, BUCK)
import blade_buckling as bb
from emit_abd import load_station_abd
import robust_checks as rc


def reduced(A):
    A = np.asarray(A, float)
    return A[0, 0] - (A[0, 1] ** 2 / A[1, 1] if abs(A[1, 1]) > 0 else 0.0)


def ring_EA(i):
    """reduced axial stiffness from the station's NATIVE ring (the truth)."""
    shell = os.path.join(bb.SHELLD, "iea_s%02d_shell.yaml" % i)
    d = yaml.safe_load(open(shell))
    nd = np.array([bb._row(n)[:2] for n in d["nodes"]])
    cells = np.array([[int(x) for x in bb._row(e)] for e in d["elements"]]); cells -= cells.min()
    name_of = {}
    for grp in d["sets"]["element"]:
        for lab in grp["labels"]:
            name_of[int(lab) - 1] = grp["name"]
    ay = load_station_abd(os.path.join(bb.ABDD, "iea_s%02d_abd.yaml" % i))["by_name"]
    tot = 0.0
    for k, c in enumerate(cells):
        ds = np.linalg.norm(nd[c[1]] - nd[c[0]])
        nm = name_of.get(k, d["sections"][0]["elementSet"])
        tot += reduced(ay[nm][0]) * ds
    return tot, len(cells)


print("station  ring_elems conf_elems |  EA_ring      EA_conf(nsub=1)  ratio | EA_conf(nsub=12)  ratio")
for i in [5, 10, 20, 30, 40]:
    EAr, nring = ring_EA(i)
    shell = os.path.join(bb.SHELLD, "iea_s%02d_shell.yaml" % i)
    r = i / 50.0
    oml = bb.resample(np.asarray(bb.build_cross_section(bb.blade, r=r)["nodes"], float), bb.N)
    P = np.zeros((bb.Ntot, 2)); P[:bb.N] = oml
    for w in range(bb.NWEB):
        Pa, Pb = oml[bb.web_ia[w]], oml[bb.web_ib[w]]
        tl = np.linspace(0, 1, bb.NW)[1:-1]
        P[bb.wnode(w, 0):bb.wnode(w, 0) + bb.NWI] = Pa[None, :] + tl[:, None] * (Pb - Pa)[None, :]
    tree, names = bb.station_layup_lookup(shell)
    ay = load_station_abd(os.path.join(bb.ABDD, "iea_s%02d_abd.yaml" % i))["by_name"]
    a, b = bb.sec_elems[:, 0], bb.sec_elems[:, 1]
    ds = np.linalg.norm(P[b] - P[a], axis=1)
    out = []
    for nsub in (1, 12):
        ABD, _ = bb.assign_abd(P, tree, names, ay, nsub=nsub)
        EAc = float(np.sum([reduced(ABD[se]) for se in range(bb.NSE)] * ds))
        out.append((EAc, EAc / EAr))
    print(" s%02d      %4d       %4d     | %.4e   %.4e  %.3f | %.4e   %.3f"
          % (i, nring, bb.NSE, EAr, out[0][0], out[0][1], out[1][0], out[1][1]))
    # guard form
    A11 = np.array([bb.assign_abd(P, tree, names, ay, nsub=12)[0][se][0, 0] for se in range(bb.NSE)])
    A12 = np.array([bb.assign_abd(P, tree, names, ay, nsub=12)[0][se][0, 1] for se in range(bb.NSE)])
    A22 = np.array([bb.assign_abd(P, tree, names, ay, nsub=12)[0][se][1, 1] for se in range(bb.NSE)])
    g = rc.abd_ea_consistency(A11, A12, A22, ds, EAr)
    print("          guard abd_ea_consistency -> %s (ratio %.4f)" % (g["verdict"], g["ratio"]))
