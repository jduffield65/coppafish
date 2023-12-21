from . import call_spots, extract, find_spots, omp, pipeline, setup, spot_colors, stitch, utils, robominnie
from .Unsure import register
from .pipeline.run import run_pipeline
from .setup import Notebook, NotebookPage
from .utils.pciseq import export_to_pciseq
from ._version import __version__
try:
    from . import plot
    from .plot import Viewer
    from.plot.pdf.base import BuildPDF
except ModuleNotFoundError:
    # So no error if not installed napari/matplotlib etc
    pass
