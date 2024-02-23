import os
from importlib.resources import files

path = __package__
if not path:
    path = 'nuthatch'

# CLI constants
MPL_STYLER = files(path).joinpath('resource/style/mplstyler.mplstyle')
TRAIN_FILE = files(path).joinpath('resource/file/billets.csv')
PLOT_PATH = str(files(path).joinpath('resource/plot'))
