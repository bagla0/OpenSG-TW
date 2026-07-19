"""emit_blade_shell.py -- ONE full-blade shell YAML: the whole IEA-22 blade as a single conformal
skin+web shell-quad model a user can load to build a shell FE / buckling model.

Contents (windIO-style, extended from the 1-D station yaml to a 3-D quad mesh):
  nodes                : (Nnode,3) blade coordinates (x=span, y=chord, z=flap)
  elements             : (Nelem,4) quad connectivity (1-indexed)
  elementOrientations  : per-element [e1(span), e2(arc), e3(outward normal)] material frame (9 floats)
  sets.element         : element labels grouped by (station,layup) key
  sections             : each key -> shell layup ply stack [[material, thickness, angle], ...] (mid-ref)
  materials            : the composite material database
  reference            : center (mid-surface)

Per-station layup lookup + span loft (constant layup between stations, from the left boundary station),
exactly the ABD-assignment convention used by blade_buckling.py.  No full-mesh reprojection."""
import os, sys, time
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BUCK)
import blade_buckling as bb                       # reuse topology, resample, station layup lookup
import yaml

N, NWEB, NWI, MPER, Ntot = bb.N, bb.NWEB, bb.NWI, bb.MPER, bb.Ntot
sec_elems = bb.sec_elems; NSE = len(sec_elems); NSTA = bb.NSTA
SHELLD = bb.SHELLD; BLADE_LEN = bb.BLADE_LEN
OUT = os.path.join(bb.DATA, "iea22_blade_shell.yaml")


def station_data(i):
    """coords P (Ntot,2), layup NAME per section element, and the ply-stack dict for station i."""
    shell = os.path.join(SHELLD, "iea_s%02d_shell.yaml" % i)
    d = yaml.safe_load(open(shell)); r = i / 50.0
    oml = bb.resample(np.asarray(bb.build_cross_section(bb.blade, r=r)["nodes"], float), N)
    P = np.zeros((Ntot, 2)); P[:N] = oml
    for w in range(NWEB):
        Pa, Pb = oml[bb.web_ia[w]], oml[bb.web_ib[w]]
        tl = np.linspace(0, 1, bb.NW)[1:-1]
        P[bb.wnode(w, 0):bb.wnode(w, 0) + NWI] = Pa[None, :] + tl[:, None] * (Pb - Pa)[None, :]
    tree, names = bb.station_layup_lookup(shell)
    mids = 0.5 * (P[sec_elems[:, 0]] + P[sec_elems[:, 1]])
    idx = tree.query(mids + (tree.data.mean(0) - mids.mean(0)))[1]     # align conformal frame -> yaml frame
    lay = [names[j] for j in idx]                                     # layup name per section element
    stacks = {s["elementSet"]: [[p[0], float(p[1]), float(p[2])] for p in s["layup"]] for s in d["sections"]}
    return P, lay, stacks, d["materials"]


def main():
    t0 = time.time()
    Pk = np.zeros((NSTA, Ntot, 2)); layk = []; stack_by_key = {}; materials = None
    for i in range(NSTA):
        P, lay, stacks, mats = station_data(i)
        Pk[i] = P; layk.append(lay); materials = mats
        for se in range(NSE):
            stack_by_key[(i, lay[se])] = stacks[lay[se]]              # (station,layup) -> ply stack
        if i % 10 == 0:
            print("  station %d built" % i)
    Rk = np.arange(NSTA) / 50.0 * BLADE_LEN
    NS = (NSTA - 1) * MPER + 1
    # nodes
    nodes = np.zeros((NS * Ntot, 3))
    for p in range(NS):
        rp = p / MPER; kL = min(int(np.floor(rp)), NSTA - 2); tt = rp - kL
        P = (1 - tt) * Pk[kL] + tt * Pk[kL + 1]; X = (1 - tt) * Rk[kL] + tt * Rk[kL + 1]
        nodes[p * Ntot:(p + 1) * Ntot, 0] = X
        nodes[p * Ntot:(p + 1) * Ntot, 1] = P[:, 0]; nodes[p * Ntot:(p + 1) * Ntot, 2] = P[:, 1]
    # elements + orientations + set membership (by the LEFT station's layup)
    quads = []; ori = []; key_of = []
    for p in range(NS - 1):
        b0, b1 = p * Ntot, (p + 1) * Ntot
        kL = min(p // MPER, NSTA - 1)                                 # left real station for this span segment
        for se in range(NSE):
            a, bb_ = sec_elems[se]
            q = [b0 + bb_, b1 + bb_, b1 + a, b0 + a]; quads.append(q)   # e1(edge0->1)=SPAN, matches solver frame
            X = nodes[q]
            v1 = X[1] - X[0]; v2 = X[3] - X[0]
            e3 = np.cross(v1, v2); e3 /= np.linalg.norm(e3) + 1e-30                        # element normal
            e1 = v1 / (np.linalg.norm(v1) + 1e-30); e2 = np.cross(e3, e1)                  # e1=span, e2=arc
            ori.append(np.r_[e1, e2, e3]); key_of.append((kL, layk[kL][se]))
    quads = np.array(quads); ori = np.array(ori)
    # dedup layup keys -> named sections
    keys = sorted(set(key_of)); kname = {k: "lay_s%02d_%s" % (k[0], k[1]) for k in keys}
    sets = {kname[k]: [] for k in keys}
    for e, k in enumerate(key_of):
        sets[kname[k]].append(e + 1)
    # write yaml (stream by hand for compact arrays)
    L = ["nodes:"]
    for nd in nodes:
        L.append("- [%.8f, %.8f, %.8f]" % (nd[0], nd[1], nd[2]))
    L.append("elements:")
    for q in quads:
        L.append("- [%d, %d, %d, %d]" % (q[0] + 1, q[1] + 1, q[2] + 1, q[3] + 1))
    L.append("elementOrientations:")
    for o in ori:
        L.append("- [" + ", ".join("%.8f" % v for v in o) + "]")
    L += ["sets:", "  element:"]
    for k in keys:
        L.append("  - name: %s" % kname[k]); L.append("    labels: [%s]" %
                 ", ".join(str(x) for x in sets[kname[k]]))
    L.append("sections:")
    for k in keys:
        L.append("- type: shell"); L.append("  elementSet: %s" % kname[k]); L.append("  layup:")
        for ply in stack_by_key[k]:
            L.append("  - [%s, %.8f, %.4f]" % (ply[0], ply[1], ply[2]))
    L.append("materials:")
    L.append(yaml.safe_dump(materials, default_flow_style=False, sort_keys=False).rstrip())
    L.append("reference: center")
    open(OUT, "w").write("\n".join(L) + "\n")
    print("\nwrote %s\n  %d nodes, %d quads, %d layup sections (%d stations x up-to-6 layups)  in %.1fs"
          % (OUT, len(nodes), len(quads), len(keys), NSTA, time.time() - t0))


if __name__ == "__main__":
    main()
