"""pvpython: render each point-data array of a .vtk to a PNG (rainbow, top view, scalar bar, no
title).  Usage:  pvpython render_pv.py <in.vtk> <outdir> <prefix> <field1> <field2> ..."""
from paraview.simple import *
import os, sys

vtkfile, outdir, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
fields = sys.argv[4:]
os.makedirs(outdir, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

reader = OpenDataFile(vtkfile)
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1700, 620]
view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 0
view.CameraParallelProjection = 1
disp = Show(reader, view)
disp.Representation = 'Surface'
UpdatePipeline()

for f in fields:
    ColorBy(disp, ('POINTS', f))
    disp.RescaleTransferFunctionToDataRange(True, False)
    lut = GetColorTransferFunction(f)
    lut.ApplyPreset('Rainbow Uniform', True)
    disp.SetScalarBarVisibility(view, True)
    sb = GetScalarBar(lut, view)
    sb.Title = f
    sb.ComponentTitle = ''
    sb.TitleColor = [0, 0, 0]; sb.LabelColor = [0, 0, 0]
    sb.TitleFontSize = 18; sb.LabelFontSize = 15
    sb.ScalarBarLength = 0.55
    ResetCamera(view)
    view.CameraParallelScale = view.CameraParallelScale * 0.62      # zoom in, trim whitespace
    Render(view)
    out = os.path.join(outdir, "%s_%s.png" % (prefix, f))
    SaveScreenshot(out, view, ImageResolution=[1500, 650], TransparentBackground=0)
    print("wrote", out)
    disp.SetScalarBarVisibility(view, False)
