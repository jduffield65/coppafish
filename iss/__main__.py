# This file allows the program to be called from the command line without
# entering the Python interpreter.  To call, use:
#
#     python3 -m iss inifile.ini

from iss import run_pipeline
import sys
import os
import textwrap

def print_usage(message = None):
    if message:
        message = f"\n\n    ERROR: {message}"
    else:
        message = ''
    USAGE = f"""
    === ISS processing software ===

    To call, pass a single argument containing the name of the config file.  E.g.,

        python3 -m iss config.ini
    {message}
    """
    exit(textwrap.dedent(USAGE))

# Ensure there is exactly one argument, and it is an ini file
if len(sys.argv) == 1:
    print_usage("Please pass the config file as an argument")
if len(sys.argv) >= 3:
    print_usage("Please only pass one config file as an argument")
if sys.argv[1] in ["--help", "-h"]:
    print_usage()
if not os.path.isfile(sys.argv[1]):
    print_usage(f"Cannot find path {sys.argv[1]}, please specify a valid file")

run_pipeline(sys.argv[1])
