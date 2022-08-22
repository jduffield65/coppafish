# Running the code

Once the configuration file has been [set up](config_setup.md) with the path 
*/Users/user/coppafish/experiment/settings.ini*, the [code](code/pipeline/run.md#coppafish.pipeline.run.run_pipeline) 
can be run via the command line or using a python script:

=== "Command Line"

    ``` bash
    python -m coppafish /Users/user/coppafish/experiment/settings.ini
    ```

=== "Python Script"
    ``` python
    from coppafish import run_pipeline
    ini_file = '/Users/user/coppafish/experiment/settings.ini'
    nb = run_pipeline(ini_file)
    ```

If the pipeline has already been partially run and the [notebook.npz](notebook.md) file exists in the output directory, 
the above will pick up the pipeline from the last stage it finished. So for a notebook that contains the pages 
[*file_names*](notebook_comments.md#file_names), [*basic_info*](notebook_comments.md#basic_info),
[*extract*](notebook_comments.md#extract), [*extract_debug*](notebook_comments.md#extract_debug) and 
[*find_spots*](notebook_comments.md#find_spots), running the above code will start the pipeline at
the [stitch stage](code/pipeline/run.md#coppafish.pipeline.run.run_stitch).

## Check data before running
The functions [`view_raw`](pipeline/extract.md#raw-data), 
[`view_filter`](pipeline/extract.md#viewer) and [`view_find_spots`](pipeline/find_spots.md#viewer) 
can be run before the *Notebook* is created if a valid configuration file is provided.

So if there is a dataset of questionable quality, it may be worth running some of these first to see if it 
looks ok. In particular, [`view_raw`](pipeline/extract.md#raw-data) may be useful for checking the correct
channels are used, or to see if specific tiles/z-planes should be 
[removed](config_setup.md#using-a-subset-of-the-raw-data).

## Re-run section
If at any stage, a section of the pipeline needs re-running, then the relevant *NotebookPage* must first be 
[removed](notebook.md#deleting-a-notebookpage) from the Notebook before the configuration file parameters for that 
section can be [altered](notebook.md#configuration-file).

??? example "Re-run `register_initial`"
    The code below illustrates how you can re-run the [`register_initial`](pipeline/register_initial.md) step
    of the pipeline with different configuration file parameters. 
    
    If the last line is uncommented, the full pipeline will be run, starting with `register_initial`, and the 
    *Notebook* will be saved as `notebook_new.npz` in the output directory.

    === "Code"

        ``` python
        from coppafish import Notebook, run_pipeline
        nb_file = '/Users/user/coppafish/experiment/notebook.npz'
        
        # Save new notebook with different name so it does not overwrite old notebook
        # Make sure notebook_name is specified in [file_names] section 
        # of settings_new.ini file to be same as name given here.
        nb_file_new = '/Users/user/coppafish/experiment/notebook_new.npz'
        ini_file_new = '/Users/user/coppafish/experiment/settings_new.ini'
    
        # config_file not given so will use last one saved to Notebook
        nb = Notebook(nb_file)
        config = nb.get_config()['register_initial']
        print('Using config file saved to notebook:')
        print(f"shift_max_range: {config['shift_max_range']}")
        print(f"shift_score_thresh_multiplier: {config['shift_score_thresh_multiplier']}")
    
        # Change register_initial
        del nb.register_initial     # delete old register_initial
        nb.save(nb_file_new)        # save Notebook with no register_initial page to new file 
                                    # so does not overwrite old Notebook
        # Load in new notebook with new config file
        nb_new = Notebook(nb_file_new, ini_file_new)
        config_new = nb_new.get_config()['register_initial']
        print(f'Using new config file {ini_file_new}:')
        print(f"shift_max_range: {config_new['shift_max_range']}")
        print(f"shift_score_thresh_multiplier: {config_new['shift_score_thresh_multiplier']}")
        # nb = run_pipeline(ini_file_new)   # Uncomment this line to run pipeline starting from
                                            # register_initial
        ```
    === "Output"

        ``` 
        Using config file saved to notebook:
        shift_max_range: [500, 500, 10]
        shift_score_thresh_multiplier: 1.5
        Using new config file /Users/user/coppafish/experiment/settings_new.ini:
        shift_max_range: [600, 600, 20]
        shift_score_thresh_multiplier: 1.2
    
        ```
    === "nb._config (saved to *Notebook*)"
    
        ``` ini
        [file_names]
        input_dir = /Users/user/coppafish/experiment1/raw
        output_dir = /Users/user/coppafish/experiment1/output
        tile_dir = /Users/user/coppafish/experiment1/tiles
        round = Exp1_r0, Exp1_r1, Exp1_r2, Exp1_r3, Exp1_r4, Exp1_r5, Exp1_r6
        anchor = Exp1_anchor
        code_book = /Users/user/coppafish/experiment1/codebook.txt
    
        [basic_info]
        is_3d = True
        anchor_channel = 4
        dapi_channel = 0
        ```
    === "*/Users/user/coppafish/experiment/settings_new.ini*"
    
        ``` ini
        [file_names]
        input_dir = /Users/user/coppafish/experiment1/raw
        output_dir = /Users/user/coppafish/experiment1/output
        tile_dir = /Users/user/coppafish/experiment1/tiles
        round = Exp1_r0, Exp1_r1, Exp1_r2, Exp1_r3, Exp1_r4, Exp1_r5, Exp1_r6
        anchor = Exp1_anchor
        code_book = /Users/user/coppafish/experiment1/codebook.txt
        notebook_name = notebook_new
    
        [basic_info]
        is_3d = True
        anchor_channel = 4
        dapi_channel = 0

        [register_initial]
        shift_max_range = 600, 600, 20
        shift_score_thresh_multiplier = 1.2
        ```

If the section that needs running is [`call_reference_spots`](code/pipeline/call_reference_spots.md), 
then the procedure is 
[slightly different](pipeline/call_reference_spots.md#re-run-call_reference_spots) because
this step adds variables to the [`ref_spots`](notebook_comments.md#ref_spots) page as well as creating the 
[`call_spots`](notebook_comments.md#call_spots) page.

## Exporting to pciSeq
To save the results of the pipeline as a .csv file which can then be plotted with pciSeq, one  of 
the following can be run (assuming path to the config file is */Users/user/coppafish/experiment/settings.ini* 
and the path to the notebook file is */Users/user/coppafish/experiment/notebook.npz*):

=== "Command Line"

    ``` bash
    python -m coppafish /Users/user/coppafish/experiment/settings.ini -export
    ```

=== "Python Script Using Config Path"
    ``` python
    from coppafish import Notebook, export_to_pciseq
    ini_file = '/Users/user/coppafish/experiment/settings.ini'
    nb = Notebook(config_file=ini_file)
    export_to_pciseq(nb)
    ```

=== "Python Script Using Notebook Path"
    ``` python
    from coppafish import Notebook, export_to_pciseq
    nb_file = '/Users/user/coppafish/experiment/notebook.npz'
    nb = Notebook(nb_file)
    export_to_pciseq(nb)
    ```

[This](code/utils/pciseq.md#coppafish.utils.pciseq.export_to_pciseq) will save a csv file in the *output_dir* for each method 
(*omp* and *ref_spots*) of finding spots and assigning genes to them.
The names of the files are specified through `config['file_names']['pciseq']`. Each file will contain:

- y - y coordinate of each spot in stitched coordinate system.
- x - x coordinate of each spot in stitched coordinate system.
- z_stack - z coordinate of each spot in stitched coordinate system (in units of z-pixels).
- Gene - Name of gene each spot was assigned to.

An example file is given [here](files/pciseq_omp.csv).

### Thresholding
Only spots which pass 
[`quality_threshold`](code/call_spots/qual_check.md#coppafish.call_spots.qual_check.quality_threshold)
are saved. This depends on parameters given in [`config['thresholds']`](config.md#thresholds).

For a [*reference*](pipeline/call_reference_spots.md) spot, $s$, to pass the thresholding, it must satisfy 
the following:

$$
\displaylines{\Delta_s > \Delta_{thresh}\\ \chi_s > \chi_{thresh}}
$$

Where:

* $\Delta_s$ is the maximum [dot product score](pipeline/call_reference_spots.md#dot-product-score) on iteration 0
  for spot $s$ across all genes (i.e. $\Delta_s = \max_g(\Delta_{s0g})$).
* $\Delta_{thresh}$ is `config['thresholds']['score_ref']`.
* $\chi_s$ is the [intensity](pipeline/call_reference_spots.md#intensity) of spot $s$.
* $\chi_{thresh}$ is `config['thresholds']['intensity']`. If this is not provided, it is set to
`nb.call_spots.gene_efficiency_intensity_thresh`.

For an [*OMP*](pipeline/omp.md) spot, $s$, to pass the thresholding, it must satisfy the following:

$$
\displaylines{\gamma_s > \gamma_{thresh}\\ \chi_s > \chi_{thresh}}
$$

Where:

* $\gamma_s$ is the [*OMP* score](pipeline/omp.md#omp-score) for spot $s$.
* $\gamma_{thresh}$ is `config['thresholds']['score_omp']`.
* $\chi_s$ and $\chi_{thresh}$ are the same as for the *reference spots*.

It is important that these thresholds are greater than 0, because when running the pipeline, we try to save a lot 
of spots. The idea being this is that it is better to do the thresholding after the pipeline has been run rather than 
during the pipeline. This is because, if there were too few spots in the latter case, much of the pipeline would
have to be re-run to obtain new spots but in the former case, you can just change the threshold values.

Once [`export_to_pciseq`](code/utils/pciseq.md#coppafish.utils.pciseq.export_to_pciseq) is run, the
[*thresholds*](notebook_comments.md#thresholds) page will be added to the notebook. This
inherits all the values from the [*thresholds*](config.md#thresholds) section of the config file, the purpose of which
is to remove the possibility of the [*thresholds*](config.md#thresholds) section in the configuration
file [being changed](notebook.md#configuration-file) once the results have been exported.