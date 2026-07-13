import os, sys, numpy as np, yaml
GLB = os.path.expanduser("~/OpenSG-TW-claude/examples/data/2d_yaml/bar_urc_glb")

def read_glb(p):
    L = [ln.split() for ln in open(p) if ln.strip()]
    # lines: 4 frame rows, then [F1 M1 M2 M3], [F2 F3], then distributed loads
    a = [float(x) for x in L[4]]; b = [float(x) for x in L[5]]
    F1, M1, M2, M3 = a; F2, F3 = b
    return np.array([F1, F2, F3, M1, M2, M3])   # VABS order

print("BAR-URC .glb loads (st0..st29), VABS order [F1,F2,F3,M1,M2,M3]:")
FFs = []
for i in range(30):
    p = os.path.join(GLB, "bar_urc-%d-t-0.in.glb" % i)
    ff = read_glb(p); FFs.append(ff)
    print("  st%-2d  F=[%.3e %.3e %.3e]  M=[%.3e %.3e %.3e]" % (i, ff[0], ff[1], ff[2], ff[3], ff[4], ff[5]))
FFs = np.array(FFs)
np.save(os.path.expanduser("~/claude_tmp/barurc_FF.npy"), FFs)
print("\nsaved barurc_FF.npy shape", FFs.shape)

# ---- IEA windIO station grid ----
sys.path.insert(0, os.path.expanduser("~/OpenSG-TW-claude/third_party/OpenSG_io"))
WIN = os.path.expanduser("~/OpenSG-TW-claude/examples/data/windio/IEA-22-280-RWT.yaml")
d = yaml.safe_load(open(WIN))

def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "grid" and isinstance(v, list) and len(v) > 3:
                print("  grid @ %s/%s  (%d):" % (path, k, len(v)), [round(x, 4) for x in v][:40])
            walk(v, path + "/" + str(k))
    elif isinstance(o, list):
        for j, v in enumerate(o[:3]):
            walk(v, path + "[%d]" % j)
print("\nwindIO grids found:")
walk(d)
