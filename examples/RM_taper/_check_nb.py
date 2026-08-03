import sys, os, inspect
CC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(CC, "opensg_jax"))
try:
    import nbformat, nbconvert
    print("NB: nbformat", nbformat.__version__, "nbconvert", nbconvert.__version__)
except Exception as e:
    print("NB MISSING:", repr(e)[:120])
try:
    from fe_jax.orient_plot import plot_orient
    print("plot_orient sig:", inspect.signature(plot_orient))
except Exception as e:
    print("plot_orient import fail:", repr(e)[:150])
