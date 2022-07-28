# This file allows the program to be called from the command line without
# entering the Python interpreter.  To call, use:
#
#     python3 -m iss inifile.ini

from iss import run_pipeline, iss_plot
from iss.setup import Notebook
import sys
import os
import textwrap
import numpy as np


def print_usage(message = None):
    if message:
        message = f"\n\n    ERROR: {message}"
    else:
        message = ''
    USAGE = f"""
    === ISS processing software ===

    To run pipeline, pass a single argument containing the name of the config file.  E.g.,

        python3 -m iss config.ini
    
    To open the results viewer, pass a second argument -view. E.g.,
    
        python3 -m iss config.ini -view
    {message}
    """
    exit(textwrap.dedent(USAGE))

# Ensure there is exactly one argument, and it is an ini file
if len(sys.argv) == 1:
    print_usage("Please pass the config file as an argument")
if len(sys.argv) >= 4:
    print_usage(f"Please only pass config file as first argument and optionally -view as the second.\n"
                f"But {len(sys.argv)-1} arguments passed:\n{sys.argv[1:]}")
if sys.argv[1] in ["--help", "-h"]:
    print_usage()
if not os.path.isfile(sys.argv[1]):
    print_usage(f"Cannot find path {sys.argv[1]}, please specify a valid file")

if len(sys.argv) == 2:
    run_pipeline(sys.argv[1])
if len(sys.argv) == 3:
    if not np.isin(sys.argv[2], ['view', '-view', '-plot', 'plot']):
        print_usage(f"To plot results, second argument should be -view but {sys.argv[2]} given")
    else:
        nb = Notebook(config_file=sys.argv[1])
        iss_plot(nb)
