# Pipeline

Here we outline in detail, how each part of the pipeline works including what each parameter in
the [configuration file](../config.md) is used for. Some useful [plots](../code/plot/viewer.md) 
that can be run for debugging purposes or to help understand what is going on are indicated. Potential
errors that can be hit during each stage of the pipeline are also mentioned.

Instructions on how to re-run a section of the pipeline are given [here](../run_code.md#re-run-section).

A brief description on what each step of the pipeline does is given below:

??? note "Times"

    The times taken for a *3D* experiment with 4 tiles, 7 rounds, 
    7 channels and tiles of dimension $2048\times 2048\times 50$ are given for each section. 
    There are three sets of profiling results:
    
    * [One](../files/timings/full_run_extract_jax.csv) for the whole pipeline, but with code that is a bit outdated.
    * [One](../files/timings/full_run_no_extract_jax.csv) with up to date code, excluding the *extract and filter* step.
    * [One](../files/timings/full_run_no_extract_no_jax.csv) with up to date code, 
    excluding the *extract and filter* step and not using the [optimised](../index.md) code.

    These times were obtained on an *M1 2021 Macbook Pro*. Because the [*nd2*](https://github.com/tlambert03/nd2) 
    library does not work on this computer, the [*nd2reader*](https://github.com/Open-Science-Tools/nd2reader) 
    library was used instead, hence the time taken for the *extract and filter* stage may not be accurate.

    Note that *nd2reader* does not work for QuadCam data, hence we use the *nd2* library.

* [*Extract and Filter*](extract.md): This loads in an image from the 
[input directory](../config_setup.md#input_dir) for each tile, 
round and channel. It then filters each one and saves them as *npy* files in the
[tile directory](../config_setup.md#tile_dir).

    *Time: 57 minutes*

* [*Find Spots*](find_spots.md): For each tile, round and channel, this loads in the filtered image
from the [tile directory](../config_setup.md#tile_dir). It then [detects spots](find_spots.md#spot-detection) 
on each one and saves the $yxz$ coordinates of each spot to the *Notebook*. This now gives us a point cloud 
for each tile, round and channel.

    *Time: 7 minutes, 10 seconds*
    <br />
    *Time (not optimised): 61 minutes, 50 seconds*

* [*Stitching*](stitch.md): This takes the point clouds on neighbouring tiles of the 
[reference round](../config_setup.md#ref_round) / [reference channel](../config_setup.md#ref_round) and uses
them to find the overlap between the tiles. After doing this for all sets of neighbouring tiles,
we obtain a $yxz$ origin coordinate (bottom left corner) for each tile such that a global coordinate system is created
(i.e. given a coordinate on a tile, we can obtain the global coordinate by adding the origin coordinate
of that tile).

    *Time: 43 seconds (14 seconds per shift)*

* [*Register Initial*](register_initial.md): For each tile, this finds the shift between
the [reference round](../config_setup.md#ref_round) / [reference channel](../config_setup.md#ref_round) and 
each sequencing round through an exhaustive search using the point clouds. 
In total, $n_{tiles} \times n_{rounds}$ shifts are found.

    *Time: 5 minutes, 48 seconds (12.4 seconds per shift)*

* [*Register*](register.md): This takes the shifts found in the [*Register Initial*](register_initial.md) step
and uses them as a [starting point](register.md#starting-transform) to determine the affine transform between the 
[reference round](../config_setup.md#ref_round) / [reference channel](../config_setup.md#ref_round) and each
sequencing round and channel for each tile. This is done through an [iterative closest point](register.md#icp) 
algorithm (*ICP*) and in total, $n_{tiles} \times n_{rounds} \times n_{channels}$ transforms are found.

    *Time: 20 seconds*

* [*Get Reference Spots*](get_reference_spots.md): For each spot detected on the 
[reference round](../config_setup.md#ref_round) / [reference channel](../config_setup.md#ref_round), 
this uses the transforms obtained in the [*Register*](register.md) step to determine the corresponding 
coordinate in each sequencing round and channel. For each sequencing round and chanel, it then loads
in the filtered image from the [tile directory](../config_setup.md#tile_dir) and reads off the 
intensity at the computed coordinate. This gives a $n_{rounds}\times n_{channels}$ 
[*spot color*](get_reference_spots.md#reading-off-intensity) for each reference spot.

    *Time: 3 minutes, 18 seconds*
    <br />
    *Time (not optimised): 3 minutes, 18 seconds*   

* [*Call Reference Spots*](call_reference_spots.md): For each gene, a $n_{rounds}\times n_{channels}$ 
[*bled code*](call_reference_spots.md#gene-bled-codes)
is obtained which indicates what the *spot color* of spots assigned to that gene should look like. For each spot,
we determine which gene it corresponds to by computing a [dot product](call_reference_spots.md#dot-product-score) 
between its *spot color* and each gene *bled code*.
The spot is assigned to the gene for which this dot product is the largest.

    *Time: 16 seconds*
    <br />
    *Time (not optimised): 30 seconds*  

* [*OMP*](omp.md): For each pixel in the global coordinates, the *spot color* is obtained. Then multiple 
gene *bled codes* are [fit](omp.md#omp-algorithm) until the residual *spot color* cannot be explained by any 
further genes. For each 
gene that is fit, we obtain a coefficient indicating how much of that gene's *bled code* is required to explain the 
*spot color*. Once this is done for all pixels, we have a coefficient for each gene at each pixel. This allows
us to build a [coefficient image](omp.md#finding-spots) for each gene. 
Spots are then [detected](find_spots.md#spot-detection) on these images.
This gives us a second distribution of genes [which allows for](omp.md#why-bother-with-omp) 
overlapping spots and spots at locations not detected in the 
[reference round](../config_setup.md#ref_round) / [reference channel](../config_setup.md#ref_round).

    *Time: 1 hour, 16 minutes*
    <br />
    *Time (not optimised): 20 hours, 53 minutes*  


*Total Time: 2 hours, 33 minutes*
<br />
*Total Time (not optimised): 23 hours, 6 minutes*

??? note "Order of Steps"

    The results of the [*Stitching*](stitch.md) part of the pipeline are first used in the 
    [*Get Reference Spots*](get_reference_spots.md) step. Thus, the [*Stitching*](stitch.md) part
    can actually be run anywhere between the [*Find Spots*](find_spots.md) and
    [*Get Reference Spots*](get_reference_spots.md) steps.

    All other steps must be run in the order indicated.
