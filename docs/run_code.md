# Running the code

Once the config file has been [set up](config_setup.md) with the path */Users/user/iss/experiment/settings.ini*, 
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

If the pipeline has already been partially run and the notebook .npz file exists in the output directory, 
the above will pick up the pipeline from the last stage it finished. So for a notebook that contains the pages 
[*file_names*](notebook_comments.md#file_names), [*basic_info*](notebook_comments.md#basic_info),
[*extract*](notebook_comments.md#extract), [*extract_debug*](notebook_comments.md#extract_debug) and 
[*find_spots*](notebook_comments.md#find_spots), running the above code will start the pipeline at
the [stitch stage](code/pipeline/run.md#iss.pipeline.run.run_stitch).
