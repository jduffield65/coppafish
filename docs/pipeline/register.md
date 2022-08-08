# Register
The [*register* step of the pipeline](../code/pipeline/register.md) finds an affine transform between
the reference round/reference channel ($r_{ref}$/$c_{ref}$) and each imaging round and channel for every tile.
The affine transform to tile $t$, round $r$, channel $c$ is found through iterative closest point
using the shift, `nb.register_initial.shift[t, r]`, found during the [*register_initial*](register_initial.md) step of 
the pipeline as the starting point. It is saved to the *Notebook* as `nb.register.transform[t, r, c]`.
These transforms are used later in the pipeline to determine the intensity of a pixel in each round and channel.

The [`register`](../notebook_comments.md#register) and [`register_debug`](../notebook_comments.md#register_debug) 
*NotebookPages* are added to the *Notebook* after this stage is finished.

## Affine Transform
We want to find the transform from tile $t$, round $r_{ref}$, channel $c_{ref}$ to round $r$, channel $c$ for all
tiles, rounds and channels such that there is pixel-level alignment between the images. The pixel-level 
alignment is important because most spots are only a few pixels in size, so even a one-pixel registration error 
can compromise the `spot_colors` found in the next stage of the pipeline and thus the gene assignment.

The shifts found in the [*register_initial*](register_initial.md) step of 
the pipeline are not sufficient for this, because of chromatic aberration which will cause a scaling between 
color channels. There may also be small rotational or non-rigid shifts; thus we find affine transformations 
which can include shifts, scalings, rotations and shears.

### Starting Transform
The affine transforms are found using the [iterative-closest point](../code/register/base.md#iss.register.base.icp) 
(*ICP*) algorithm. This is highly sensitive to local maxima, so it is initialized with the shifts found in the 
[*register_initial*](register_initial.md) step, `nb.register_initial.shift`.

The starting transform to a particular round and tile is the same for all channels. The shifts are put into
the form of an affine transform ($4 \times 3$ array) through 
[`transform_from_scale_shift`](../code/register/base.md#iss.register.base.transform_from_scale_shift).

??? example "Starting transform from shifts"
    The code below indicates how the initial shifts ($n_{tiles} \times n_{rounds} \times 3$ array) 
    are converted into initial transforms ($n_{tiles} \times n_{rounds} \times n_{channels} \times 4 \times 3$ array).
    
    === "Code"
        ``` python
        import numpy as np
        from iss.register.base import transform_from_scale_shift
        t_print = 1
        r_print = 1
        initial_shifts = nb.register_initial.shift   # z shift is in z-pixel units
        print(f"Initial shift for tile {t_print}, round {r_print}:\n{initial_shifts[t_print, r_print]}")

        # Convert z shift into same units as yx pixels
        initial_shifts = initial_shifts.astype(float)
        z_scale = [1, 1, nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy]
        for t in nb.basic_info.use_tiles:
            for r in nb.basic_info.use_rounds:
                initial_shifts[t, r] = initial_shifts[t, r] * z_scale
        print(f"Initial shift for tile {t_print}, round {r_print} (z shift in YX pixels):\n"
              f"{np.around(initial_shifts[t_print, r_print], 2)}")
        
        # Convert shifts to affine transform 4 x 3 form
        initial_scale = np.ones((nb.basic_info.n_channels))  # set all scalings to 1 initially 
                                                             # as just want shift.
        initial_transforms = transform_from_scale_shift(initial_scale, initial_shifts)
        
        # Show transform different for rounds but same for channel within round
        for r in range(2):
            for c in range(2):
                print(f"Initial transform for tile {t_print}, round {r}, channel {c}:\n"
                      f"{np.around(initial_transforms[t_print, r, c], 2)}")
        ```
    === "Output"
        ``` 
        Initial Shifts for tile 1, round 1:
        [25 14  1]
        Initial shift for tile 1, round 1 (z shift in YX pixels):
        [25.   14.    5.99]
        Initial transform for tile 1, round 0, channel 0:
        [[ 1.  0.  0.]
         [ 0.  1.  0.]
         [ 0.  0.  1.]
         [48. 30.  0.]]
        Initial transform for tile 1, round 0, channel 1:
        [[ 1.  0.  0.]
         [ 0.  1.  0.]
         [ 0.  0.  1.]
         [48. 30.  0.]]
        Initial transform for tile 1, round 1, channel 0:
        [[ 1.    0.    0.  ]
         [ 0.    1.    0.  ]
         [ 0.    0.    1.  ]
         [25.   14.    5.99]]
        Initial transform for tile 1, round 1, channel 1:
        [[ 1.    0.    0.  ]
         [ 0.    1.    0.  ]
         [ 0.    0.    1.  ]
         [25.   14.    5.99]]
        ```

### *ICP*
The pseudocode for the [*ICP*](../code/register/base.md#iss.register.base.icp) algorithm to 
find the affine transform for tile $t$ round $r$, channel $c$ is indicated below.
The shape of the various arrays are indicated in the comments (#).

??? note "Preparing point clouds"

    Prior to the *ICP* algorithm, the $yxz$ coordinates of spots on a tile
    are centered. This is because, it makes more sense to me if any rotation is applied around 
    the centre of the tile.

    Also, like for the initial shifts, the z-coordinates must be converted into units of yx-pixels,
    so overall:

    ``` python
    spot_yxz = spot_yxz - nb.basic_info.tile_centre  # center coordinates
    # Convert units of z-coordinates
    z_scale = [1, 1, nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy]
    spot_yxz = spot_yxz * z_scale
    ```
    
    Also, for the non-reference point clouds we only keep spots which are isolated. This 
    is because the *ICP* algorithm is less likely to fall into local maxima if 
    the spots are quite well separated.
    
    We deam a spot [isolated](../code/find_spots/base.md#iss.find_spots.base.get_isolated_points) 
    if the nearest spot to it is further away than `2 * neighb_dist`.
    Where `neighb_dist = config['register']['neighb_dist_thresh_2d']` if *2D* and
    `config['register']['neighb_dist_thresh_3d']` if *3D*. 

`n_iter` is the maximum number of iterations and is set by `config['register']['n_iter']`. 
`neighb_dist_thresh` is the distance in $yx$ pixels below which neighbours are a good match. 
It is given by `config['register']['neighb_dist_thresh_2d']` if *2D* and
`config['register']['neighb_dist_thresh_3d']` if *3D*. 
Only neighbours which are closer than this are used when computing the transform.

??? note "Padding `ref_spot_yxz`"

    `ref_spot_yxz` has shape `[n_ref x 4]` not `[n_ref x 3]` because to be able to be multiplied by 
    the affine transform (`[4 x 3]`), it must be padded with ones i.e. `ref_spot_yxz[:, 3] = 1`.
    
    `ref_spot_yxz @ transform` is then the same as `ref_spot_yxz[:, :3] @ transform[:3, :] + transform[3]` i.e.
    rotation/scaling matrix applied and then the shift is added.

```
spot_yxz = yxz coordinates for spots detected on tile t,
           round r, channel c.  # [n_base x 3]
ref_spot_yxz = padded yxz coordinates for spots detected on tile t,
               reference round, reference channel.  # [n_ref x 4]                    
transform_initial = starting transform for tile t, round r, channel c.

transform = transform_initial  # [4 x 3]
neighb_ind = zeros(n_ref)  # [n_ref]
neighb_ind_last = neighb_ind  # [n_ref]
i = 0
while i < n_iter:
    Transform ref_spot_yxz according to transform to give 
              ref_spot_yxz_transform. # [n_ref x 3]
              This is just matrix multiplication.
              
    neighb_ind[s] is index of spot in spot_yxz closest to 
        ref_spot_yxz_transform[s].
        
    dist[s] is the distance between spot_yxz[neighb_ind[s]]
        and ref_spot_yxz_transform[s]  # [n_ref]
    
    Choose which spots to use to update the transform:
        use = dist < neighb_dist_thresh
        spot_yxz_use = spot_yxz[neighb_ind[use]]  # [n_use x 3]
        ref_spot_yxz_use = ref_spot_yxz[use]  # [n_use x 4]
    
    Update transform to be the 4x3 matrix which multiplies ref_spot_yxz_use
        such that the distances between the transformed spots 
        and spot_yxz_use are minimised. This is the least squares solution.
        
    If neighb_ind and neighb_ind_last are identical, then stop iteration
        i.e. i = n_iter + 5.
    
    neighb_ind_last = neighb_ind
    i = i + 1
```

### Checking *ICP* results
Once the [*ICP*](#icp) algorithm has obtained a transform, `transform[t, r, c]` for each tile, round and channel, 
we want to determine if they are reasonable. Two criteria must be satisfied for 
`transform[t, r, c]` to be considered reasonable:

* [Number of neighbours exceeds threshold](#number-of-neighbours)
* [`transform[t, r, c]` is near to what we broadly expect it should be](#Expected-Transform).

#### Number of neighbours


#### Expected Transform





### Regularized *ICP*


## Debugging
### [`view_icp`](../code/plot/register.md#view_icp)

