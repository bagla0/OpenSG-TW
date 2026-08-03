import numpy as np

z = np.load("data/final_iml_fields_ARCHIVED.npz")
print("IML C6 diag:", np.array2string(np.diag(z["C6"]), precision=4))
