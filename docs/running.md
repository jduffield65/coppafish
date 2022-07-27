# Setting up the Config File

A config (.ini) file needs to be created for each experiment to run the pipeline. 
All parameters not specified in this file will inherit the [default values](config.md).
The parameters with Default = `MUST BE SPECIFIED` are the bare minimum parameters which must be specified.
Some example config files for typical experiments are listed below.

## Example Config Files

=== "3D"

    ``` ini
    [file_names]
    input_dir = /Users/.../experiment1/raw
    output_dir = /Users/.../experiment1/output
    tile_dir = /Users/.../experiment1/tiles
    round = Exp1_r0, Exp1_r1, Exp1_r2, Exp1_r3, Exp1_r4, Exp1_r5, Exp1_r6
    anchor = Exp1_anchor
    code_book = /Users/.../experiment1/codebook.txt

    [basic_info]
    is_3d = True
    anchor_channel = 4
    dapi_channel = 0
    ```

=== "2D"

    ``` ini
    [file_names]
    input_dir = /Users/.../experiment1/raw
    output_dir = /Users/.../experiment1/output
    tile_dir = /Users/.../experiment1/tiles
    round = Exp1_r0, Exp1_r1, Exp1_r2, Exp1_r3, Exp1_r4, Exp1_r5, Exp1_r6
    anchor = Exp1_anchor
    code_book = /Users/.../experiment1/codebook.txt

    [basic_info]
    is_3d = False
    anchor_channel = 4
    dapi_channel = 0
    ```

=== ".npy Raw Data"

    ``` ini
    [file_names]
    input_dir = /Users/.../experiment1/raw
    output_dir = /Users/.../experiment1/output
    tile_dir = /Users/.../experiment1/tiles
    round = Exp1_r0, Exp1_r1, Exp1_r2, Exp1_r3, Exp1_r4, Exp1_r5, Exp1_r6
    anchor = Exp1_anchor
    code_book = /Users/.../experiment1/codebook.txt
    raw_extension = .npy
    raw_metadata = metadata

    [basic_info]
    is_3d = True
    anchor_channel = 4
    dapi_channel = 0
    ```

=== "No Anchor"

    ``` ini
    [file_names]
    input_dir = /Users/.../experiment1/raw
    output_dir = /Users/.../experiment1/output
    tile_dir = /Users/.../experiment1/tiles
    round = Exp1_r0, Exp1_r1, Exp1_r2, Exp1_r3, Exp1_r4, Exp1_r5, Exp1_r6
    code_book = /Users/.../experiment1/codebook.txt

    [basic_info]
    is_3d = True
    ref_round = 2
    ref_channel = 4
    ```

=== "Separate Round"

    ``` ini
    [file_names]
    notebook_name = sep_round_notebook
    input_dir = /Users/.../experiment1/raw
    output_dir = /Users/.../experiment1/output
    tile_dir = /Users/.../experiment1/tiles
    anchor = Exp1_sep_round
    code_book = /Users/.../experiment1/codebook.txt

    [basic_info]
    is_3d = True
    anchor_channel = 4
    ```

=== "QuadCam"

    ``` ini
    [file_names]
    input_dir = /Users/.../experiment1/raw
    output_dir = /Users/.../experiment1/output
    tile_dir = /Users/.../experiment1/tiles
    round = Exp1_r0, Exp1_r1, Exp1_r2, Exp1_r3, Exp1_r4, Exp1_r5, Exp1_r6, Exp1_r7, Exp1_r8
    anchor = Exp1_anchor
    code_book = /Users/.../experiment1/codebook.txt

    [basic_info]
    is_3d = True
    anchor_channel = 4
    dapi_channel = 0
    dye_names = DY405, CF405L, AF488, DY520XL, AF532, AF594, ATTO425, AF647, AF750
    channel_camera = 405, 555, 470, 470, 555, 640, 555, 640, 640
    channel_laser = 405, 405, 445, 470, 520, 520, 555, 640, 730
    ```

These variables are explained below and [here](config.md). 

## file_names
### input_dir

The input directory is the path to the folder which contains the raw data. 
Examples for the two possible cases of `raw_extension` are given below (i.e. these are respectively what the input 
directory looks like for the config files *3D* and *.npy Raw Data* listed above).
=== ".nd2"
    ![image](images/config/InputDirectory_nd2.png){width="400"}

=== ".npy"
    ![image](images/config/InputDirectory_npy.png){width="400"}

???note "Differences with `raw_extension = .npy`"

    It is assumed that when `raw_extension = .npy`, there were initial .nd2 files which contained excess information 
    (e.g. extra channels). These were then converted to .npy files to get rid of this.

    For the .npy case, input_dir must also contain a metadata .json file. 
    This contains the metadata extracted from the initial .nd2 files using the function 
    [save_metadata](code/utils/nd2.md#iss.utils.nd2.save_metadata).
    An example metadata file is given [here](images/config/metadata_example.json) for an experiment with 
    3 tiles and 7 channels.

    Also, each name listed in the `round` parameter indicates a folder not a file. 
    It is assumed these folders were produced using 
    [`dask.array.to_npy_stack`](https://docs.dask.org/en/stable/generated/dask.array.to_npy_stack.html)
    so the contents of each file should contain a file named *info* and a .npy file for each tile, with the name 
    being the index of the tile in the initial .nd2 file. An example showing the folder for the first round of a 
    three tile experiment is given below:

    ![image](images/config/InputDirectory_npy_tile_files.png){width="400"}
    


### output_dir
The output directory is the path to the folder that you would like the notebook.npz file containing the experiment 
results to be saved. The image below shows what the output directory typically looks like at the end of the experiment.

![image](images/config/OutputDirectory.png){width="400" align=left }

The names of the files produced can be changed by changing the parameters `big_anchor_image`, `big_dapi_image`, 
`notebook_name`, `omp_spot_coef`, `omp_spot_info` and `omp_spot_shape` in the [config file](config.md#file_names). 

<br clear="left"/>

### tile_dir
The tile directory is the path to the folder that you would like the filtered images for each tile, round and 
colour channel to be saved to. 

If `is_3d == True`, a .npy file will be produced for each round, tile and channel with
the name for round r, tile *T*, channel *C* being `file_names[round][r]`_t*T*c*C*.npy with axis in the order z-y-x
(the name for the anchor round, tile *T*, channel *C* will be `file_names[anchor]`_t*T*c*C*.npy).
If `is_3d == False`, a .npy file will be produced for each round and tile called `file_names[round][r]`_t*T*.npy 
with axis in the order c-y-x. 

An example of what the tile directory looks like at the end of the experiment is shown below for a 
3D and 2D experiment with 3 tiles and 7 channels:
=== "3D"
    ![image](images/config/TileDirectory3D.png){width="400"}

=== "2D"
    ![image](images/config/TileDirectory2D.png){width="400"}


### code_book
