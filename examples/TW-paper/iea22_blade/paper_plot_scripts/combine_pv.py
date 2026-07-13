"""Combine the individual ParaView-rendered PNGs (VABS | OpenSG RM) into one comparison image per
component (solid elements incl. webs)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
PV = os.path.expanduser("~/OpenSG-TW-claude/examples/TW-paper/xsec_paper/figures/pv")
FIG = os.path.expanduser("~/OpenSG-TW-claude/examples/TW-paper/xsec_paper/figures")
FIELDS = [("u1", "u_1"), ("u2", "u_2"), ("u3", "u_3"),
          ("S11", "\\sigma_{11}"), ("S22", "\\sigma_{22}"), ("S12", "\\sigma_{12}")]
for key, lab in FIELDS:
    v = mpimg.imread(os.path.join(PV, "sec_VABS_%s.png" % key))
    r = mpimg.imread(os.path.join(PV, "sec_OpenSG_RM_%s.png" % key))
    fig, ax = plt.subplots(1, 2, figsize=(15, 3.4))
    ax[0].imshow(v); ax[0].set_title(r"$%s$ -- VABS (solid)" % lab, fontsize=12)
    ax[1].imshow(r); ax[1].set_title(r"$%s$ -- OpenSG RM" % lab, fontsize=12)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    out = os.path.join(FIG, "pv_r020_%s.png" % key)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out))
