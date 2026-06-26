"""
Compare MSG-TW (JAX) vs FEniCS-solid 3D stress at the OML, leading-edge to
trailing-edge.  Reads the two files produced by ``benchmark_oml_jax.py`` and the
WSL FEniCS run, matches each JAX OML node to the nearest solid node by (y2,y3)
(KDTree), and prints a per-component comparison with percent differences.

Both stress sets are in the GLOBAL/beam frame, Voigt order [S11,S22,S33,S23,S13,S12].
"""
import os
import numpy as np
import yaml
from collections import defaultdict
from scipy.spatial import cKDTree

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
SOLID = r"C:\Users\bagla0\OpenSG\examples\data\Solid_2DSG\2Dsolid_0.yaml"
COMP = ["S11", "S22", "S33", "S23", "S13", "S12"]


def load(path):
    d = np.loadtxt(path)
    return d[:, :2], d[:, 2:]            # (N,2) coords, (N,6) stress


def solid_oml_coords(path):
    """(M,2) coordinates of the OUTER (OML) boundary loop of the solid mesh."""
    def _row(r):
        if isinstance(r, str):
            return r.strip('[]').split()
        if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
            return r[0].strip('[]').split()
        return [str(v) for v in r]
    with open(path) as f:
        vd = yaml.safe_load(f)
    vN = np.array([[float(v) for v in _row(n)] for n in vd['nodes']])[:, :2]
    elems = [[int(v) for v in _row(e)] for e in vd['elements']]
    ec = defaultdict(int); ed = {}
    for e in elems:
        k = len(e)
        for i in range(k):
            a, b = e[i] - 1, e[(i + 1) % k] - 1
            key = tuple(sorted((a, b))); ec[key] += 1; ed[key] = (a, b)
    adj = defaultdict(list)
    for key, c in ec.items():
        if c == 1:
            a, b = ed[key]; adj[a].append(b); adj[b].append(a)
    loops, seen = [], set()
    for start in list(adj):
        if start in seen:
            continue
        loop = [start]; seen.add(start); cur, prev = start, None
        while True:
            nxts = [n for n in adj[cur] if n != prev and (n not in seen or n == start)]
            if not nxts or nxts[0] == start:
                break
            cur, prev = nxts[0], cur; loop.append(cur); seen.add(cur)
        loops.append(loop)
    area = lambda lp: 0.5 * abs(np.dot(vN[lp][:, 0], np.roll(vN[lp][:, 1], -1))
                                - np.dot(vN[lp][:, 1], np.roll(vN[lp][:, 0], -1)))
    loops.sort(key=area, reverse=True)
    return vN[np.array(loops[0])]        # largest-area loop = OML


def upper_te_to_le(xy):
    """Indices of the UPPER-surface OML nodes, ordered trailing-edge -> leading-edge.

    Upper surface = the side of the LE-TE chord with the larger mean y3; ordered
    by descending y2 (chordwise), which sweeps TE (max y2) -> LE (min y2).  This
    is the same arc the plate homogenization / solid traverse, so the local
    (e1=beam, e2=arc tangent, e3=normal) orientation is consistent.
    """
    le = int(np.argmin(xy[:, 0])); te = int(np.argmax(xy[:, 0]))
    chord = xy[te] - xy[le]
    cross = chord[0] * (xy[:, 1] - xy[le, 1]) - chord[1] * (xy[:, 0] - xy[le, 0])
    side = cross >= 0
    if xy[side, 1].mean() < xy[~side, 1].mean():     # pick the y3-positive side
        side = ~side
    up = np.where(side)[0]
    return up[np.argsort(-xy[up, 0])]                # TE (max y2) -> LE (min y2)


def match_outer(jxy, fxy, fs, oml, radius=0.04):
    """For each JAX OML node pick the OUTERMOST FEniCS point within ``radius``
    (= the true outer-surface point), falling back to the nearest.  This avoids
    grabbing an inner-layer / adjacent-ply point at the thin leading edge."""
    cen = oml.mean(axis=0)
    rad_f = np.hypot(fxy[:, 0] - cen[0], fxy[:, 1] - cen[1])
    tree = cKDTree(fxy)
    out = np.zeros((len(jxy), fs.shape[1])); dist = np.zeros(len(jxy))
    for i, p in enumerate(jxy):
        cand = tree.query_ball_point(p, radius)
        if cand:
            j = cand[int(np.argmax(rad_f[cand]))]    # outermost candidate
        else:
            j = tree.query(p)[1]
        out[i] = fs[j]; dist[i] = np.hypot(*(fxy[j] - p))
    return out, dist


def compare(name, jxy, js, fpath, oml):
    fxy, fs = load(fpath)
    near = cKDTree(oml).query(fxy)[0] < 0.05            # keep near-OML points
    fxy, fs = fxy[near], fs[near]
    fs_m, dist = match_outer(jxy, fxy, fs, oml)
    smax = np.max(np.abs(np.concatenate([js, fs_m])))
    print(f"\n================  JAX-TW  vs  FEniCS-solid [{name}]  ================")
    print(f"matched {len(jxy)} OML nodes | match dist mean={dist.mean():.4f} "
          f"max={dist.max():.4f} | stress scale max|sigma|={smax:.3e} Pa")
    print(f"  {'comp':5s} {'median|d|/sc':>12s} {'p90|d|/sc':>10s} {'max|d|/sc':>10s} "
          f"{'corr':>7s}  worst@node")
    for j, c in enumerate(COMP):
        ad = np.abs(js[:, j] - fs_m[:, j]) / smax
        cc = np.corrcoef(js[:, j], fs_m[:, j])[0, 1] if np.std(fs_m[:, j]) > 1e-9 else np.nan
        print(f"  {c:5s} {np.median(ad):12.4f} {np.percentile(ad,90):10.4f} "
              f"{ad.max():10.4f} {cc:7.3f}  #{int(np.argmax(ad))}")
    return fs_m


def main():
    jxy, js = load(os.path.join(OUT, "oml_jax.txt"))
    oml = solid_oml_coords(SOLID)

    # benchmark over the UPPER surface only, trailing-edge -> leading-edge
    idx = upper_te_to_le(jxy)
    jxy, js = jxy[idx], js[idx]
    print(f"UPPER surface, TE->LE: {len(jxy)} OML nodes "
          f"(y2 {jxy[0,0]:.2f} -> {jxy[-1,0]:.2f})")

    fg = compare("Gauss q2", jxy, js, os.path.join(OUT, "oml_fenics_gauss.txt"), oml)
    compare("CG2", jxy, js, os.path.join(OUT, "oml_fenics_cg2.txt"), oml)

    # ---- TE->LE table (dominant components) for the Gauss-q2 recovery ----
    show = ["S11", "S13", "S12"]
    sidx = [COMP.index(s) for s in show]
    print("\nTE->LE upper-surface table (Pa);  J=JAX-TW  F=FEniCS Gauss-q2  "
          "d%=100*(J-F)/max|S_comp|")
    print("  idx     y2       y3   " + "".join(
        f"|{s}:    J          F      d%" for s in show))
    for i in range(len(jxy)):
        row = f"  {i:3d} {jxy[i,0]:8.3f} {jxy[i,1]:8.3f}  "
        for k in sidx:
            sc = max(np.max(np.abs(js[:, k])), 1e-30)
            row += f"| {js[i,k]:10.2e} {fg[i,k]:10.2e} {(js[i,k]-fg[i,k])/sc*100:6.1f}"
        print(row)


if __name__ == "__main__":
    main()
