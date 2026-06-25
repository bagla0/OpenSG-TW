"""% error of the 1D-shell Timoshenko 6x6 vs the 2D-solid (= VABS) reference, for every stiffness
term, at OML (frac 0) and CENTER (frac 0.5) references, across the wall-thickness sweep."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FACTORS = [0.2, 0.4, 0.6, 0.8, 1.0]
DIAG = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
LBL = ["ext", "sh2", "sh3", "tw", "b2", "b3"]


def tag(f):
    return "f%03d" % int(round(f * 100))


def load(kind, f):
    p = os.path.join(RES, "C6_%s_%s.txt" % (kind, tag(f)))
    return np.loadtxt(p) if os.path.exists(p) else None


def pe(a, b):
    return 100.0 * (a - b) / b if abs(b) > 1.0 else float("nan")


fok = [f for f in FACTORS if load("solid", f) is not None and load("shell_oml", f) is not None]
sol = {f: load("solid", f) for f in fok}
oml = {f: load("shell_oml", f) for f in fok}
cen = {f: load("shell_center", f) for f in fok}

out = []
out.append("=" * 78)
out.append("DIAGONAL stiffness %% error  (shell - solid)/solid   [OML | center]")
out.append("=" * 78)
hdr = "%-5s |" % "term" + "".join(" f=%.1f          " % f for f in fok)
out.append(hdr)
out.append("%-5s |" % "" + "".join("  OML    cen   " for _ in fok))
for i in range(6):
    row = "%-5s |" % DIAG[i]
    for f in fok:
        eo = pe(oml[f][i, i], sol[f][i, i]); ec = pe(cen[f][i, i], sol[f][i, i])
        row += " %+6.1f %+6.1f |" % (eo, ec)
    out.append(row)

out.append("")
out.append("=" * 78)
out.append("OFF-DIAGONAL couplings %% error  (terms with |solid| > 1e6 at f=1.0)   [OML | center]")
out.append("=" * 78)
out.append(hdr)
out.append("%-5s |" % "" + "".join("  OML    cen   " for _ in fok))
ref = sol[fok[-1]]
for i in range(6):
    for j in range(i + 1, 6):
        if abs(ref[i, j]) > 1e6:
            row = "%s-%s |" % (LBL[i], LBL[j])
            row = "%-5s |" % ("%s-%s" % (LBL[i], LBL[j]))
            for f in fok:
                eo = pe(oml[f][i, j], sol[f][i, j]); ec = pe(cen[f][i, j], sol[f][i, j])
                row += " %+6.0f %+6.0f |" % (eo, ec)
            out.append(row)

txt = "\n".join(out)
print(txt)
open(os.path.join(RES, "error_table.txt"), "w").write(txt + "\n")
print("\nwrote results/error_table.txt")
