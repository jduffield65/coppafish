# coppafish

![https://github.com/reillytilbury/coppafish/blob/dev/.github/workflows/continuous_integration.yaml](https://github.com/reillytilbury/coppafish/actions/workflows/continuous_integration.yaml/badge.svg)

![](https://github.com/jduffield65/coppafish/blob/main/docs/images/readme_viewer.png?raw=true)

*coppafish* is a data analysis pipeline for decoding *coppaFISH* (combinatorial padlock-probe-amplified fluorescence in 
situ hybridization) datasets. [*coppaFISH*](https://www.nature.com/articles/s41586-022-04915-7) 
is a method for in situ transcriptomics which produces a series of images arranged in terms of tiles, rounds
and channels. *coppafish* then determines the distribution of genes via 
[image processing](https://jduffield65.github.io/coppafish/pipeline/extract/), 
[spot detection](https://jduffield65.github.io/coppafish/pipeline/find_spots/), 
[registration](https://jduffield65.github.io/coppafish/pipeline/register/) and 
[gene calling](https://jduffield65.github.io/coppafish/pipeline/call_reference_spots/).


## Prerequisites
Python 3.9 or 3.10

[Git](https://git-scm.com/) (optional).


## Installation

Like many python programs, coppafish requires specific versions of library packages.  To use coppafish without altering
your existing python installation, create an environment specifically for it using a tool like [anaconda](https://www.anaconda.com/download).  
To create a conda environment for coppafish and activate it, type these commands into a terminal:
```console
conda create -n coppafish python=3.9
condate activate coppafish
```

Next, go to the directory where you want to store the coppafish code. 
To download the latest alpha version and install all required packages into your conda directory, type:

```console
git clone --depth 1 https://github.com/reillytilbury/coppafish
pip install -r ./coppafish/requirements.txt
pip install ./coppafish[plotting,optimised]
```

After installing, you can run the code by typing ```python -m coppafish experiment.ini```, where ```experiment.ini``` 
should be replaced by the name of the configuration file for your experiment.

An alternative way of installing is to run the coppafish code from the directory in which you downloaded it.  This allows you
to edit the code and checkout different versions from github, so is more suitable for people who want the latest versions. To do this, type
```console
git clone https://github.com/reillytilbury/coppafish
pip install -r ./coppafish/requirements.txt
pip install -e ./coppafish[plotting,optimised]
```


## Documentation
For more information on usage, please see the 
[documentation](https://jduffield65.github.io/coppafish/).

