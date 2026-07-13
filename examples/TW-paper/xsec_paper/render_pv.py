"""pvpython: render each point-data array of a .vtk to a PNG -- rainbow, top view, its OWN scalar
bar (no title text), white background.  Usage:
    pvpython render_pv.py <in.vtk> <outdir> <prefix> <field1> <field2> ...
The exploded mesh carries POINT data, so ParaView interpolates WITHIN each element while the stress
stays discontinuous across material (web/skin) boundaries (VABS-style, no junction bleeding)."""
from paraview.simple import *
import os, sys

vtkfile, outdir, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
fields = sys.argv[4:]
os.makedirs(outdir, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

reader = OpenDataFile(vtkfile)
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1700, 640]
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
    sb.Title = ''                                  # separate per-panel bar, but NO written title text
    sb.ComponentTitle = ''
    sb.TitleColor = [0, 0, 0]; sb.LabelColor = [0, 0, 0]
    sb.LabelFontSize = 15
    sb.ScalarBarLength = 0.55
    ResetCamera(view)
    view.CameraParallelScale = view.CameraParallelScale * 0.62
    Render(view)
    out = os.path.join(outdir, "%s_%s.png" % (prefix, f))
    SaveScreenshot(out, view, ImageResolution=[1500, 650], TransparentBackground=0)
    print("wrote", out)
    disp.SetScalarBarVisibility(view, False)
