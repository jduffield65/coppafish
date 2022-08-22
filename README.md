# coppafish

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
Python >= 3.8

## Documentation
For more information on installation and usage, please see the 
[documentation](https://jduffield65.github.io/coppafish/).

