import importlib, sys
print("python:", sys.executable)
for m in ("dolfinx", "slepc4py", "petsc4py", "gmsh", "ufl", "basix", "mpi4py"):
    try:
        mod = importlib.import_module(m)
        print("  %-10s OK  %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("  %-10s --  %s" % (m, type(e).__name__))
