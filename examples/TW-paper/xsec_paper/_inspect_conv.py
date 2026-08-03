import numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "results", "ex3_iea_conv.npz"))
print("keys:", list(z.keys()))
nn = np.asarray(z["nnode"]).astype(int)
print("nnode:", np.sort(nn).tolist())
print("diag_err shape:", np.asarray(z["diag_err"]).shape)
o = np.argsort(nn)
err = np.asarray(z["diag_err"])[o]
for i, n in enumerate(np.sort(nn)):
    print(n, np.array2string(err[i], precision=2, floatmode="fixed"))
