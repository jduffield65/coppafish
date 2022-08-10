# iss_python
 Python version of iss software

It requires Python 3.8 and software that needs to be installed is indicated in the files SoftwareInstallWindows.txt or 
SoftwareInstallMax.txt.

Then to make sure it is working, run test.py which should take about 5 minutes and have no failures or errors.

To run an experiment, create an .ini file with all the required experiment details (see iss/setup/settings.default.ini
for parameters that need/can be set) and then change ini_file in main.py to this file path.
Then run main.py.

An example experiment .ini file is given as experiment_settings_example.ini. 
This is for a 9 rounds x 9 channels QuadCam3D experiment.

To run the software with config file "config_file.ini" from the command line, do:

> python -m iss config_file.ini


Website: https://jduffield65.github.io/iss_python/config_setup/
