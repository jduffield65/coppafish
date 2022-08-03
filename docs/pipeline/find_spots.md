# Find Spots

The [find spots step of the pipeline](../code/pipeline/find_spots.md) loads in the filtered images for each 
tile, round, channel saved during the [extract step](extract.md) and detects spots on them. 
It is obtaining a point cloud from the images because in the stitch and register sections of the pipeline,
it is quicker to use point clouds than the full images.

The [`find_spots`](../notebook_comments.md#find_spots) *NotebookPages* is added to the *Notebook* after this stage
is finished.

## Spot detection
The spots on tile $t$, round $r$, channel $c$ [are](../code/find_spots/detect.md#iss.find_spots.detect.detect_spots) 
the local maxima in the filtered image (loaded in through 
[`load_tile(nb.file_names, nb.basic_info, t, r, c)`](../code/utils/npy.md#iss.utils.npy.load_tile)) 
with an intensity greater than [`auto_thresh[t, r, c]`](extract.md#auto_thresh). 

Local maxima means pixel with the largest intensity in a neighbourhood defined by `config['find_spots']['radius_xy']`
and `config['find_spots']['radius_z']` (`kernel = np.ones((2*radius_xy-1, 2*radius_xy-1, 2*radius_z-1))`). 
The position of the local maxima is found to be where the 
[dilation](../code/utils/morphology.md#iss.utils.morphology.base.dilate) of the image with the `kernel` 
is equal to the image.

??? note "Optimised spot detection"

    The dilation method is quite slow so if *jax* is installed, a 
    [different spot detection method](../code/find_spots/detect.md#iss.find_spots.detect_optimised.detect_spots) 
    is used.

    In this method, we look at all pixels with intensity greater than `auto_thresh[t, r, c]`. For each of these,
    we say that the pixel is a spot if it has a greater intensity than all of its neighbouring pixels, where the 
    neighbourhood is determined by the `kernel`. 

    The larger `auto_thresh[t, r, c]` and the smaller the `kernel`, the faster this method is, whereas the value
    of `auto_thresh[t, r, c]` makes no difference to the speed of the dilation method. In our case, 
    `auto_thresh[t, r, c]` is pretty large as the whole point is that all background pixels (the vast majority) 
    have intensity less than it.

    TODO: GIVE TIMES FOR THIS SECTION WITH THE TWO DIFFERENT METHODS

??? note "Dealing with duplicates"

    If there are two neighbouring pixels which have the same intensity which is the local maxima intensity, by default
    both pixels will be declared to be local maxima. However if `remove_duplicates == True` in 
    [`detect_spots`](../code/find_spots/detect.md#iss.find_spots.detect.detect_spots), only one will be deemed a 
    local maxima.

    This is achieved by adding a random shift to the intensity of each pixel. The max possible shift is 0.2 so it 
    will not change the integer version of the image but it will ensure each pixel has a different intensity to 
    its neighbour.

### Imaging spots
For non reference spots (all round/channel combinations apart from `ref_round`/`ref_channel`),
we only use the spots for registration to the reference spots so the quantity of
spots is not important. In fact, registration tends to work better if there are fewer but more reliable spots
as this means there is a lesser chance of matching up spots by chance. 

To exploit this, for each imaging tile $t$, round $r$, channel $c$ where $r$, the point cloud 
is made up of the `max_spots` most intense spots on each z-plane. In *2D*, `max_spots` is 
`config['find_spots']['max_spots_2d']` and in *3D*, it is `config['find_spots']['max_spots_3d']`.
If there are fewer than `max_spots` spots detected on a particular z-plane, all the spots will be kept.


### Reference spots
We want to assign a gene to each reference spot (`ref_round`/`ref_channel`) as well as use it for registration,
so it is beneficial to maximise the number of reference spots. As such, we do not do the `max_spots` thresholding
we do for [imaging spots](#imaging-spots).

However, we want to know which reference spots are isolated because when it comes to the 
[`bleed_matrix`](../code/call_spots/bleed_matrix.md#iss.call_spots.bleed_matrix.get_bleed_matrix) calculation,
we do not want to use overlapping spots.

#### Isolated spots
We deem a spot to be [isolated](../code/find_spots/base.md#iss.find_spots.base.get_isolated) 
if it has a prominent [negative annulus](extract.md#effect-of-filtering),
because if there was an overlapping spot, you would expect positive intensity in the annulus around the spot.
We find the intensity of the annulus by computing the correlation of the image with an 
[annulus kernel](../code/utils/strel.md#iss.utils.strel.annulus) where:

* `r0 = config['find_spots']['isolation_radius_inner']`
* `r_xy = config['find_spots']['isolation_radius_xy']` 
* `r_z = config['find_spots']['isolation_radius_z']`.

If the value of this correlation at the location of a spot is less than `config['find_spots']['isolation_thresh']`,
then we deem the spot to be isolated. If `config['find_spots']['isolation_thresh']` is not given, it is set to
`config['find_spots']['auto_isolation_thresh_multiplier'] * auto_thresh[t, r, c]`. 
The final isolation thresholds used for each round, tile, channel are saved as 
[nb.find_spots.isolation_thresh](../notebook_comments.md#find_spots).

## Viewer
### Negative neighbour