"""Emit per-station pipeline status as CSV: station,eta,xml,shell1d,sg,solid2d,rm_out,jax_out."""
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def has(pattern):
    return 1 if glob.glob(os.path.join(HERE, pattern)) else 0


print("station,eta,xml,shell1d,sg,solid2d,rm6x6,jax6x6,rm_out,jax_out")
for i in range(51):
    s = "s%02d" % i
    eta = i / 50.0
    row = [s, "%.3f" % eta,
           has("xml/iea_%s.xml" % s),
           has("1d_yaml/iea_%s*.yaml" % s),
           has("sg/iea_%s.sg" % s),
           has("2d_yaml/iea_%s*.yaml" % s),
           has("homo_rm/*%s*" % s) or has("out/OpenSG_RM_Shell/*%s*" % s),
           has("homo_jax/*%s*" % s) or has("out/OpenSG_JAX_Solid/*%s*" % s),
           has("out/OpenSG_RM_Shell/*%s*" % s),
           has("out/OpenSG_JAX_Solid/*%s*" % s)]
    print(",".join(map(str, row)))
