# Get Reference Spots
The [*get reference spots* step of the pipeline](../code/pipeline/get_reference_spots.md) uses the transform
found in the [register](register.md) step of the pipeline to compute the corresponding coordinate of each
reference spot (detected on the reference round/reference channel ($r_{ref}$/$c_{ref}$) in 
[*find spots* step](find_spots.md)) in each imaging round and channel. 
By reading off the intensity values at these coordinates, an $n_{rounds} \times n_{channels}$ 
intensity vector can be found for each reference spot. 

These intensity vectors are saved as `colors` in the [`ref_spots`](../notebook_comments.md#ref_spots) *NotebookPage* 
which is added to the *Notebook* after this stage is finished and are used for assigning each spot to a gene
in the call_reference_spots step. 

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.



## Project layout
### Hello

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

[Link_to_NotebookPage](code/setup/notebook.md#notebook-page)

[Link_to_optimised-get_local_maxima_jax](code/find_spots/detect.md#iss.find_spots.detect_optimised.get_local_maxima_jax)