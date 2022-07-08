from iss.plot.results_viewer.base import iss_plot
from iss.setup import Notebook

# If ini_file = None, omp diagnostic won't work but others should.
ini_file = '/Users/.../experiment/settings.ini'
nb_file = '/Users/.../experiment/notebook.npz'
nb = Notebook(nb_file, ini_file)
iss_plot(nb)

## Diagnostics
# Bleed Matrix: Press b key
# Bled Codes for All Genes: Press g key - Then scroll with mouse to change gene
# Spot Color and Bled Code Matched to: Select spot of interest in select_points mode then press c key
# Spot intensity in all rounds/channels: Select spot of interest in select_points mode then press s key
# OMP coefficients in neighbourhood of spot: Select spot of interest in select_points mode then press o key
