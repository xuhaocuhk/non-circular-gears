from debug_util import MyDebugger
from plot.qt_plot import Plotter
from models import Model, find_model_by_name
import shape_factory
import os

files = os.listdir(r'C:\Projects\gears\silhouette')
new_models = []
for file in files:
    if '.png' in file or '.jpg' in file:
        new_models.append(file[:-4])

debugger = MyDebugger('print_input')
plotter = Plotter()

for model in new_models:
    drive_model = find_model_by_name(model)
    drive_smooth = shape_factory.get_shape_contour(drive_model, uniform=True, plots=None, smooth=32)
    drive_nmooth = shape_factory.get_shape_contour(drive_model, uniform=True, plots=None, smooth=0)
    plotter.draw_contours(debugger.file_path(f'{model}_smooth.png'), [('input_drive', drive_smooth)], [])
    plotter.draw_contours(debugger.file_path(f'{model}_nmooth.png'), [('input_drive', drive_nmooth)], [])
