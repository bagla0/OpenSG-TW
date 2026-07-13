import os, numpy as np, shutil
FF = np.load(os.path.expanduser("~/claude_tmp/barurc_FF.npy"))    # (30,6) [F1,F2,F3,M1,M2,M3]
s_bar = np.arange(30) / 29.0                                       # BAR-URC normalized span
D2 = os.path.expanduser("~/OpenSG-TW-claude/examples/data/2d_yaml")
SP = os.path.expanduser("~/claude_tmp/span_out")
stations = [0.2470, 0.3993, 0.5336, 0.7389, 0.9800]
names = ["iea_r0247", "iea_r0399", "iea_r0534", "iea_r0739", "iea_r0980"]

print("Spanwise-varying FF (BAR-URC distribution mapped by normalized span), VABS order:")
print("  %-11s %6s | %10s %10s %10s %10s %10s %10s" % ("station", "r", "F1", "F2", "F3", "M1", "M2", "M3"))
rows = []
for r, nm in zip(stations, names):
    ff = np.array([np.interp(r, s_bar, FF[:, k]) for k in range(6)])
    rows.append((r, ff))
    glb = ("0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
           "%.10g %.10g %.10g %.10g\n%.10g %.10g\n" % (ff[0], ff[3], ff[4], ff[5], ff[1], ff[2])
           + "0 0 0 0 0 0\n" * 4)
    pv = os.path.join(SP, nm, nm + "_prevabs")
    open(os.path.join(pv, nm + ".sg.glb"), "w").write(glb)
    for ext in ("sg", "sg.glb", "sg.mat"):
        src = os.path.join(pv, "%s.%s" % (nm, ext))
        if os.path.exists(src):
            shutil.copy(src, os.path.join(D2, "%s.%s" % (nm, ext)))
    print("  %-11s %6.3f | %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e" % (nm, r, *ff))
np.save(os.path.join(D2, "span_FF.npy"), np.array([r for r, _ in rows] + [0]))  # marker
print("\nstaged iea_r0247/0399/0534/0739/0980  .sg + .sg.glb + .sg.mat  in 2d_yaml")
print("run each locally:  vabs iea_rXXXX.sg 10   ->  .SM + .U")
