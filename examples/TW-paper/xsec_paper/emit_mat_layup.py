"""emit_mat_layup.py -- KL-paper-style material-properties and layup tables for the
IEA-22 r/R=0.2 cross-section, read from the actual shell_r020.yaml used in the paper.
  -> results/tex_rm/iea_mat.tex, results/tex_rm/iea_layup_r020.tex
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
IB = os.path.abspath(os.path.join(HERE, "..", "iea22_blade", "data"))
TEX = os.path.join(HERE, "results", "tex_rm"); os.makedirs(TEX, exist_ok=True)

d = yaml.safe_load(open(os.path.join(IB, "shell_r020.yaml")))
NAME = {"gelcoat": "Gelcoat", "glass_triax": "Glass triax", "glass_uniax": "Glass uniax",
        "glass_biax": "Glass biax", "carbon_uniax": "Carbon uniax",
        "medium_density_foam": "Foam (MD)"}
REGION = {0: "Glass-uniax skin", 1: "Foam-cored panel", 2: "Carbon spar cap",
          3: "Glass-uniax skin", 4: "Carbon spar cap", 5: "Shear web"}

# ---------------- materials table ----------------
lines = [r"\begin{table}[htpb]", r"\centering",
         r"\caption{Constituent material properties of the IEA-22-280 blade laminates "
         r"($E$, $G$ in GPa; $\rho$ in kg/m$^3$).}", r"\label{tab:iea_mat}",
         r"\setlength{\tabcolsep}{5pt}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lcccccccccc}", r"\hline",
         r"Material & $E_1$ & $E_2$ & $E_3$ & $G_{12}$ & $G_{13}$ & $G_{23}$ & "
         r"$\nu_{12}$ & $\nu_{13}$ & $\nu_{23}$ & $\rho$ \\", r"\hline"]
for mm in d["materials"]:
    e = mm.get("elastic", mm)
    E = [v / 1e9 for v in e["E"]]; G = [v / 1e9 for v in e["G"]]; nu = e["nu"]
    rho = mm.get("density", 0.0)
    lines.append("%-14s & %.3g & %.3g & %.3g & %.4g & %.4g & %.4g & %.3g & %.3g & %.3g & %.0f \\\\"
                 % (NAME.get(mm["name"], mm["name"]), E[0], E[1], E[2], G[0], G[1], G[2],
                    nu[0], nu[1], nu[2], rho))
lines += [r"\hline", r"\end{tabular}%", r"}", r"\end{table}"]
open(os.path.join(TEX, "iea_mat.tex"), "w").write("\n".join(lines))
print("wrote iea_mat.tex (%d materials)" % len(d["materials"]))

# ---------------- layup table (group consecutive identical plies) ----------------
lines = [r"\begin{table}[htpb]", r"\centering",
         r"\caption{Wall laminates (layups) of the $r/R=0.2$ section, grouped by structural "
         r"region and listed from the outer mould line inward. Each row is one stack of "
         r"identical plies; the layer thickness is the ply thickness times the number of "
         r"plies.}", r"\label{tab:iea_layup}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llclccc}", r"\hline",
         r"Name & Region & Layer & Material & Ply thickness (m) & Orientation ($^\circ$) & "
         r"Number of plies \\", r"\hline"]
for si, sec in enumerate(d["sections"]):
    plies = [(p[0], float(p[1]), float(p[2])) for p in sec["layup"]]
    groups = []
    for p in plies:
        if groups and groups[-1][0] == p[0] and abs(groups[-1][2] - p[2]) < 1e-9 \
                and abs(groups[-1][1] - p[1]) < 1e-12:
            groups[-1][3] += 1
        else:
            groups.append([p[0], p[1], p[2], 1])
    nm = sec.get("elementSet", "layup_%d" % si).replace("_", r"\_")
    for k, (mn, t, ang, np_) in enumerate(groups):
        head = (r"\multirow{%d}{*}{%s} & \multirow{%d}{*}{%s}" % (len(groups), nm,
                len(groups), REGION.get(si, "")) if k == 0 else " & ")
        lines.append("%s & %d & %s & %.4g & %g & %d \\\\"
                     % (head, k + 1, NAME.get(mn, mn), t, ang, np_))
    lines.append(r"\hline")
lines += [r"\end{tabular}%", r"}", r"\end{table}"]
open(os.path.join(TEX, "iea_layup_r020.tex"), "w").write("\n".join(lines))
print("wrote iea_layup_r020.tex (%d layups)" % len(d["sections"]))
