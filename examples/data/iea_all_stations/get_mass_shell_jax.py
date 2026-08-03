"""get_mass_shell_jax.py -- JAX/numpy port of opensg get_mass_shell (1-D shell ring
mass -> beam 6x6), plus a wall-normal-projection FIX, validated against the VABS .K
6x6 mass for iea_r0020.

FAITHFUL port  = exactly reproduces opensg.utils/core shell.get_mass_shell:
   per-element through-thickness moments (mu, mx3, i22) integrated over the mid-surface
   contour, treating the through-thickness direction as the GLOBAL x3 axis.
CORRECTED      = projects the through-thickness moments onto the actual WALL NORMAL
   n=(n2,n3)=elementOrientation e3, i.e. a wall point sits at (x2+z*n2, x3+z*n3).

Per-layup through-thickness moments about the laminate MID-surface:
   mu  = sum_plies rho*t
   mx3 = sum_plies rho*t*z_mid                       (z from mid-surface, +ve OML->IML)
   i22 = sum_plies rho*(t*z_mid^2 + t^3/12)

Run:  python get_mass_shell_jax.py
"""
import os
import numpy as np
import yaml

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    HAVE_JAX = True
except Exception:
    jnp = np
    HAVE_JAX = False

HERE = os.path.dirname(os.path.abspath(__file__))
YAML = os.path.join(HERE, "1d_yaml", "iea_r0020_shell.yaml")
KFILE = os.path.join(HERE, "sg", "iea_r0020.sg.K")


# ---------------------------------------------------------------- yaml parsing
def _row(r):
    if isinstance(r, list):
        r = r[0] if (len(r) == 1 and isinstance(r[0], str)) else r
    if isinstance(r, str):
        return [float(v) for v in r.replace(",", " ").split()]
    return [float(v) for v in r]


def load_shell(path):
    d = yaml.safe_load(open(path))
    rx = np.array([_row(r)[:3] for r in d["nodes"]], dtype=float)
    cells = np.array([[int(v) for v in _row(e)] for e in d["elements"]], dtype=int)
    if cells.min() == 1:
        cells = cells - 1
    ori = np.array([_row(o) for o in d["elementOrientations"]], dtype=float)
    re3 = ori[:, 6:9]                              # WALL NORMAL per element
    sections = d["sections"]
    materials = d["materials"]
    rho_by_name = {m["name"]: float(m["density"]) for m in materials}
    setname_to_sec = {s["elementSet"]: i for i, s in enumerate(sections)}
    rsub = np.zeros(len(cells), dtype=int)
    for grp in d["sets"]["element"]:
        si = setname_to_sec[grp["name"]]
        for lab in grp["labels"]:
            rsub[int(lab) - 1] = si
    # per-section (layup) through-thickness moments about the mid-surface
    mom = np.zeros((len(sections), 3))             # (mu, mx3, i22)
    for si, sec in enumerate(sections):
        layup = sec["layup"]
        th = np.array([float(p[1]) for p in layup])
        rho = np.array([rho_by_name[p[0]] for p in layup])
        T = th.sum()
        z_bot = np.concatenate([[0.0], np.cumsum(th)])[:-1]   # OML face of each ply
        z_mid = z_bot + 0.5 * th - 0.5 * T                    # ply centre from mid-surface
        mu = np.sum(rho * th)
        mx3 = np.sum(rho * th * z_mid)
        i22 = np.sum(rho * (th * z_mid**2 + th**3 / 12.0))
        mom[si] = (mu, mx3, i22)
    return dict(rx=rx, cells=cells, rsub=rsub, re3=re3, mom=mom, cross=[0, 1])


# ---------------------------------------------------------------- .K parsing
def parse_K_mass(path):
    lines = open(path).read().splitlines()
    for i, ln in enumerate(lines):
        if "6X6 Mass Matrix" in ln and "at the" not in ln and "Shear" not in ln:
            j = i + 1
            while "====" not in lines[j]:
                j += 1
            j += 1
            rows = []
            k = j
            while len(rows) < 6 and k < len(lines):
                toks = lines[k].split()
                vals = []
                for t in toks:
                    try:
                        vals.append(float(t))
                    except ValueError:
                        pass
                if vals:
                    rows.append(vals)
                k += 1
            # each matrix row may wrap onto 2 physical lines -> regroup into 6 numbers
            flat = [v for r in rows for v in r]
            return np.array(flat[:36]).reshape(6, 6)
    raise RuntimeError("mass matrix not found")


# ---------------------------------------------------------------- integration
def _gauss_ds(rx, cells, cross):
    a = rx[cells[:, 0]][:, cross]
    b = rx[cells[:, 1]][:, cross]
    ds = np.linalg.norm(b - a, axis=1)                 # element length
    g = 1.0 / np.sqrt(3.0)
    xi = np.array([-g, g])
    # x at the 2 Gauss points (E,2gp,2coord)
    xg = 0.5 * (1 - xi)[None, :, None] * a[:, None, :] + 0.5 * (1 + xi)[None, :, None] * b[:, None, :]
    return xg, ds


def get_mass_shell_faithful(S):
    xg, ds = _gauss_ds(S["rx"], S["cells"], S["cross"])
    x2 = jnp.asarray(xg[:, :, 0]); x3 = jnp.asarray(xg[:, :, 1])
    mu = jnp.asarray(S["mom"][S["rsub"], 0])[:, None]
    mx3 = jnp.asarray(S["mom"][S["rsub"], 1])[:, None]
    i22 = jnp.asarray(S["mom"][S["rsub"], 2])[:, None]
    w = jnp.asarray(ds)[:, None] * 0.5                 # 2-pt Gauss weight*ds/2

    def I(f):
        return jnp.sum(jnp.broadcast_to(f, x2.shape) * w)

    M11 = I(mu)
    M15 = I(mx3 + x3 * mu)
    M16 = I(-x2 * mu)
    M44 = I(i22 + 2 * x3 * mx3 + mu * x2**2 + x3**2 * mu)
    M55 = I(i22 + 2 * x3 * mx3 + x3**2 * mu)
    M66 = I(mu * x2**2)
    M56 = I(-x2 * (mx3 + x3 * mu))
    return _pack(M11, M15, M16, M44, M55, M66, M56)


def get_mass_shell_corrected(S):
    xg, ds = _gauss_ds(S["rx"], S["cells"], S["cross"])
    x2 = jnp.asarray(xg[:, :, 0]); x3 = jnp.asarray(xg[:, :, 1])
    ne = np.arange(len(S["cells"]))
    n2 = jnp.asarray(S["re3"][ne, S["cross"][0]])[:, None]
    n3 = jnp.asarray(S["re3"][ne, S["cross"][1]])[:, None]
    mu = jnp.asarray(S["mom"][S["rsub"], 0])[:, None]
    mx3 = jnp.asarray(S["mom"][S["rsub"], 1])[:, None]
    i22 = jnp.asarray(S["mom"][S["rsub"], 2])[:, None]
    w = jnp.asarray(ds)[:, None] * 0.5

    def I(f):
        return jnp.sum(jnp.broadcast_to(f, x2.shape) * w)

    M11 = I(mu)
    M15 = I(x3 * mu + n3 * mx3)
    M16 = I(-(x2 * mu + n2 * mx3))
    M66 = I(x2**2 * mu + 2 * x2 * n2 * mx3 + n2**2 * i22)
    M55 = I(x3**2 * mu + 2 * x3 * n3 * mx3 + n3**2 * i22)
    M56 = I(x2 * x3 * mu + (x2 * n3 + x3 * n2) * mx3 + n2 * n3 * i22)
    M44 = M55 + M66
    return _pack(M11, M15, M16, M44, M55, M66, M56)


def _pack(M11, M15, M16, M44, M55, M66, M56):
    M11, M15, M16, M44, M55, M66, M56 = [float(v) for v in (M11, M15, M16, M44, M55, M66, M56)]
    return np.array([
        (M11,   0,   0,    0,  M15,  M16),
        (0,   M11,   0, -M15,    0,    0),
        (0,     0, M11, -M16,    0,    0),
        (0,  -M15,-M16,  M44,    0,    0),
        (M15,   0,   0,    0,  M55,  M56),
        (M16,   0,   0,    0,  M56,  M66)])


def cmp(name, M, ref):
    print("\n==== %s vs VABS .K (term-by-term %%err) ====" % name)
    idx = [(0, 0, "M11"), (0, 4, "M15"), (0, 5, "M16"),
           (3, 3, "M44"), (4, 4, "M55"), (5, 5, "M66"), (4, 5, "M56")]
    for i, j, lbl in idx:
        s = M[i, j]; r = ref[i, j]
        e = 100.0 * (s - r) / r if abs(r) > 1e-9 else float("nan")
        print("  %-4s shell=%+.6e  .K=%+.6e  err=%+8.2f%%" % (lbl, s, r, e))


if __name__ == "__main__":
    print("JAX available:", HAVE_JAX)
    S = load_shell(YAML)
    print("nodes=%d  elements=%d  sections=%d" % (len(S["rx"]), len(S["cells"]), len(S["mom"])))
    print("per-layup (mu, mx3, i22):")
    for si in range(len(S["mom"])):
        print("  layup_%d: mu=%.4f  mx3=%.6e  i22=%.6e" % (si, *S["mom"][si]))

    # ---- debug integration consistency
    xg, ds = _gauss_ds(S["rx"], S["cells"], S["cross"])
    mu_e = S["mom"][S["rsub"], 0]
    x2mid = xg[:, :, 0].mean(1)
    perim = ds.sum()
    Mtot = np.sum(mu_e * ds)
    print("\n[dbg] n_elem=%d  perimeter=%.4f  sum(mu*ds)=%.3f" % (len(ds), perim, Mtot))
    print("[dbg] x2 range [%.4f, %.4f]  weighted mean x2=%.4f" %
          (x2mid.min(), x2mid.max(), np.sum(x2mid * mu_e * ds) / Mtot))
    print("[dbg] rsub counts:", {int(k): int((S["rsub"] == k).sum()) for k in range(len(S["mom"]))})
    print("[dbg] ds min/max/median: %.5f %.5f %.5f" % (ds.min(), ds.max(), np.median(ds)))
    ml = np.argsort(ds)[-5:]
    print("[dbg] longest elems (node pairs):", [tuple(S["cells"][i]) for i in ml], "ds=", ds[ml])

    ref = parse_K_mass(KFILE)
    print("\nVABS .K 6x6 mass:")
    for r in ref:
        print("  " + "  ".join("%+.6e" % v for v in r))

    Mf = get_mass_shell_faithful(S)
    print("\nFAITHFUL JAX shell 6x6 mass:")
    for r in Mf:
        print("  " + "  ".join("%+.6e" % v for v in r))
    cmp("FAITHFUL", Mf, ref)

    Mc = get_mass_shell_corrected(S)
    print("\nCORRECTED (wall-normal) JAX shell 6x6 mass:")
    for r in Mc:
        print("  " + "  ".join("%+.6e" % v for v in r))
    cmp("CORRECTED", Mc, ref)

    # mass center check
    print("\nmass center faithful:  x2=%.5f x3=%.5f" % (-Mf[0, 5] / Mf[0, 0], Mf[0, 4] / Mf[0, 0]))
    print("mass center corrected: x2=%.5f x3=%.5f" % (-Mc[0, 5] / Mc[0, 0], Mc[0, 4] / Mc[0, 0]))
    print("mass center VABS .K:   x2=%.5f x3=%.5f" % (-ref[0, 5] / ref[0, 0], ref[0, 4] / ref[0, 0]))
