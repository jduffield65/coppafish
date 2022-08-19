# Running the code

Once the configuration file has been [set up](config_setup.md) with the path */Users/user/iss/experiment/settings.ini*, 
the [code](code/pipeline/run.md#iss.pipeline.run.run_pipeline) can be run via the command line or using a python script:

=== "Command Line"

    ``` bash
    python -m iss /Users/user/iss/experiment/settings.ini
    ```

=== "Python Script"
    ``` python
    from iss import run_pipeline
    ini_file = '/Users/user/iss/experiment/settings.ini'
    nb = run_pipeline(ini_file)
    ```

If the pipeline has already been partially run and the [notebook.npz](notebook.md) file exists in the output directory, 
the above will pick up the pipeline from the last stage it finished. So for a notebook that contains the pages 
[*file_names*](notebook_comments.md#file_names), [*basic_info*](notebook_comments.md#basic_info),
[*extract*](notebook_comments.md#extract), [*extract_debug*](notebook_comments.md#extract_debug) and 
[*find_spots*](notebook_comments.md#find_spots), running the above code will start the pipeline at
the [stitch stage](code/pipeline/run.md#iss.pipeline.run.run_stitch).

## Check data before running


## Re-run section

## Exporting to pciSeq
To save the results of the pipeline as a .csv file which can then be plotted with pciSeq, one  of 
the following can be run (assuming path to the config file is */Users/user/iss/experiment/settings.ini* 
and the path to the notebook file is */Users/user/iss/experiment/notebook.npz*):

=== "Command Line"

    ``` bash
    python -m iss /Users/user/iss/experiment/settings.ini -export
    ```

=== "Python Script Using Config Path"
    ``` python
    from iss import Notebook, export_to_pciseq
    ini_file = '/Users/user/iss/experiment/settings.ini'
    nb = Notebook(config_file=ini_file)
    export_to_pciseq(nb)
    ```

=== "Python Script Using Notebook Path"
    ``` python
    from iss import Notebook, export_to_pciseq
    nb_file = '/Users/user/iss/experiment/notebook.npz'
    nb = Notebook(nb_file)
    export_to_pciseq(nb)
    ```

[This](code/utils/pciseq.md#iss.utils.pciseq.export_to_pciseq) will save a csv file in the *output_dir* for each method 
(*omp* and *ref_spots*) of finding spots and assigning genes to them.
The names of the files are specified through `config['file_names']['pciseq']`. Each file will contain:

- y - y coordinate of each spot in stitched coordinate system.
- x - x coordinate of each spot in stitched coordinate system.
- z_stack - z coordinate of each spot in stitched coordinate system (in units of z-pixels).
- Gene - Name of gene each spot was assigned to.

Only spots which pass [`quality_threshold`](code/call_spots/qual_check.md#iss.call_spots.qual_check.quality_threshold)
are saved. This depends on parameters given in [`config['thresholds']`](config.md#thresholds).

An example file is given [here](files/pciseq_omp.csv).

This will also add a page named [*thresholds*](notebook_comments.md#thresholds) to the notebook which 
inherits all the values from the [*thresholds*](config.md#thresholds) section of the config file. 
This is to remove the possibility of the [*thresholds*](config.md#thresholds) section in the config
file [being changed](notebook.md#configuration-file) once the results have been exported.