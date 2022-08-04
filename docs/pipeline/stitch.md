# Stitch
The [stitch step of the pipeline](../code/pipeline/stitch.md) uses the reference point clouds
(all tiles of `ref_round`/`ref_channel`) added to the *Notebook* during the [`find_spots`](find_spots.md) step
to find the overlap between neighbouring tiles in the form of shifts. It then uses these shifts to get
the origin of each tile in a global coordinate system.

The [`stitch`](../notebook_comments.md#stitch) *NotebookPage* is added to the *Notebook* after this stage
is finished.

## Shift
We need to find the overlap between each pair of neighbouring tiles. To do this, for each tile, we
ask whether there is a tile to the north of it and if there is we compute the shift between the two.
We then ask if there is a tile to the east of it and if there is we compute the shift between the two.

??? example

    For a $2\times3$ ($n_y \times n_x$) grid of tiles, the indices are:

    | 2  | 1  | 0  |

    | 5  | 4  | 3  |

    We consider each tile in turn:
    
    * Tile 0 has no tiles to the north or east so we go to tile 1.
    * Tile 1 has a tile to the east (0) so we find the shift between tile 1 and tile 0.
    * Tile 2 has a tile to the east (1) so we find the shift between tile 2 and tile 1.
    * Tile 3 has a tile to the north (0) so we find the shift between tile 3 and tile 0.
    * Tile 4 has a tile to the north (1) so we find the shift between tile 4 and tile 1.
    Tile 4 also has a tile to the east (3) so we find the shift between tile 4 and tile 3.
    * Tile 5 has a tile to the north (2) so we find the shift between tile 5 and tile 2.
    Tile 5 also has a tile to the east (4) so we find the shift between tile 5 and tile 4.

### Initial range
We compute the shift through an exhaustive search in a given range. The initial range used for a tile to the north 
can be specified through `config['stitch']['shift_south_min']` and `config['stitch']['shift_south_max']`.
The range used for a tile to the east
can be specified through `config['stitch']['shift_west_min']` and `config['stitch']['shift_west_max']`.

??? note "Confusion between north/south and east/west"

    For finding the shift to a tile in the north, the parameters used in the config file and saved to the 
    *NotebookPage* have the `south` prefix. This is because if tile B is to the north of tile A, the shift applied
    to tile A to get the correct overlap is to the south (i.e. negative in the y direction).

    Equally, for finding the shift to a tile in the east, the parameters used in the config file and saved to the 
    *NotebookPage* have the `west` prefix. This is because if tile B is to the east of tile A, the shift applied
    to tile A to get the correct overlap is to the west (i.e. negative in the x direction).

The range in the $i$ direction will then be between `shift_min[i]` and `shift_max[i]` with a spacing of 
`config['stitch']['shift_step'][i]`.

If these are left blank, the range will be computed automatically using `config['stitch']['expected_overlap']`
and `config['stitch']['auto_n_shifts']`.

??? example "Example automatic range calculation"

    For an experiment with 
    
    * `nb.basic_info.tile_sz = 2048`
    * `config['stitch']['expected_overlap'] = 0.1`
    * `config['stitch']['auto_n_shifts'] = 20, 20, 1`
    * `config['stitch']['shift_step'] = 5, 5, 3`

    the range used for a tile to the north is [computed](../code/stitch/starting_shifts.md) as follows:
    === "Code"
        ``` python
        import numpy as np
        expected_overlap = config['stitch']['expected_overlap']
        auto_n_shifts = config['stitch']['auto_n_shifts']
        shift_step = config['stitch']['shift_step']
        
        expected_shift = np.array([-(1 - expected_overlap]) * [tile_sz, 0, 0]).astype(int)
        print(f"Expected shift = {expected_shift}")
        range_extent = auto_n_shifts * shift_step
        print(f"YXZ range extent: {range_extent}")  
        range_min = expected_shift - range_extent
        range_max = expected_shift + range_extent
        print(f"YXZ range min: {range_min}")
        print(f"YXZ range max: {range_max}")
        shifts_y = np.arange(range_min[0], range_max[0] + shift_step[0]/2, shift_step[0])
        print(f"Y exhaustive search shifts:\n{shifts_y}")
        shifts_x = np.arange(range_min[1], range_max[1] + shift_step[1]/2, shift_step[1])
        print(f"X exhaustive search shifts:\n{shifts_x}")
        shifts_z = np.arange(range_min[2], range_max[2] + shift_step[2]/2, shift_step[2])
        print(f"Z exhaustive search shifts:\n{shifts_z}")
        ```
    === "Output"
        ```
        Expected shift = [-1843     0     0]
        YXZ range extent: [100 100   3]
        YXZ range min: [-1943  -100    -3]
        YXZ range max: [-1743   100     3]
        Y exhaustive search shifts:
        [-1943 -1938 -1933 -1928 -1923 -1918 -1913 -1908 -1903 -1898 -1893 -1888
         -1883 -1878 -1873 -1868 -1863 -1858 -1853 -1848 -1843 -1838 -1833 -1828
         -1823 -1818 -1813 -1808 -1803 -1798 -1793 -1788 -1783 -1778 -1773 -1768
         -1763 -1758 -1753 -1748 -1743]
        X exhaustive search shifts:
        [-100  -95  -90  -85  -80  -75  -70  -65  -60  -55  -50  -45  -40  -35
          -30  -25  -20  -15  -10   -5    0    5   10   15   20   25   30   35
           40   45   50   55   60   65   70   75   80   85   90   95  100]
        Z exhaustive search shifts:
        [-3  0  3]
        ```

    For a tile to the east, the calculation is exactly the same except
    `expected shift = [0     -1843     0]`.

### Obtaining best shift
Here is some pseudocode for how we 
[obtain the best shift](../code/stitch/shift.md#iss.stitch.shift.get_best_shift_3d) between tile 5
and tile 4 from an exhaustive search. The comments (#) give the shape of the indicated array.

``` 
function find_neighbour_distances(yxz_0, yxz_1):
    # yxz_0: [n_spots_0 x 3]
    # yxz_1: [n_spots_1 x 3]
    For i in range(n_spots_0):
        find nearest spot in yxz_1 to yxz_0[i] to be the one at index j.
        distances[i] = distance between yxz_0[i] and yxz_1[j].
    return distances  # [n_spots_0]
    
function get_score(distances, dist_thresh):
    # distances: [n_spots]
    This is a function that basically counts the number of values in 
    distances which are below dist_thresh. 
    I.e. the more close neighbours, the better the shift and thus the 
    score should be larger.
    The function used in the pipeline returns the float given by: 
        score = sum(exp(-distances ** 2 / (2 * dist_thresh ** 2)))
        If all values in distances where 0 (perfect), score = n_spots.
        If all values in distances where infinity or much larger than 
        dist_thresh (bad), score = 0.
    
 
# tile_5_yxz: [n_spots_t5 x 3]
# tile_4_yxz: [n_spots_t4 x 3]
# exhaustive_search: [n_shifts x 3]

for shift in exhaustive_search:
    tile_5_yxz_shifted = tile_5_yxz + shift   # [n_spots_t5 x 3]
    distances = find_neighbour_distances(tile_5_yxz_shifted, 
                                         tile_4_yxz)  # [n_spots_t5]
    score = get_score(distances, dist_thresh)  # float
best_shift is the shift with the best score
```

In the [score function](../code/stitch/shift.md#iss.stitch.shift.shift_score), the dist_thresh parameter 
thus specifies the distance below which neighbours are a good match. It is specified through
`config['stitch']['neighb_dist_thresh']`.


#### 3D
For speed, rather than considering an exhaustive search in three dimensions, we first ignore any shift in z and 
just find the [best yx shift](../code/stitch/shift.md#iss.stitch.shift.get_best_shift_2d).

To do this, we [split](../code/stitch/shift.md#iss.stitch.shift.get_2d_slices) 
each *3D* point cloud into a number of *2D* point clouds. 
The number is determined by `config['stitch']['nz_collapse']` to be:

`ceil(nb.basic_info.nz / config['stitch']['nz_collapse'])` 

We then consider the corresponding point clouds independently.

??? example

    Lets consider finding the best yx shift between tile 5 and tile 4 with:

    * `nb.basic_info.nz = 50`
    * `config['stitch']['nz_collapse'] = 30`
    * `config['stitch']['neighb_dist_thresh'] = 2`

    The pseudocode is:
    
    ```
    # tile_5_yxz: [n_spots_t5 x 3]
    # tile_4_yxz: [n_spots_t4 x 3]
    # exhaustive_search_yx: [n_shifts_yx x 2]

    n_2d_point_clouds = ceil(50 / 30) = 2 so we need 2 2D point clouds
    split tile_5_yxz into tile_5A_yx and tile_5B_yx:
        - tile_5A_yx are the yx coordinates of every spot in 
          tile_5_yxz with z coordinate between 0 and 24 inclusive.
          # [n_spots_t5A x 2]
        - tile_5B_yx are the yx coordinates of every spot in 
          tile_5_yxz with z coordinate between 25 and 49 inclusive.
          # [n_spots_t5B x 2]
    split tile_4_yxz into tile_4A_yx and tile_4B_yx:
        - tile_4A_yx are the yx coordinates of every spot in 
          tile_4_yxz with z coordinate between 0 and 24 inclusive.
          # [n_spots_t4A x 2]
        - tile_4B_yx are the yx coordinates of every spot in 
          tile_4_yxz with z coordinate between 25 and 49 inclusive.
          # [n_spots_t4B x 2]

    for shift_yx in exhaustive_search_yx:
        tile_5A_yx_shifted = tile_5A_yx + shift_yx   # [n_spots_t5A x 2]
        distancesA = find_neighbour_distances(tile_5A_yx_shifted, 
                                              tile_4A_yx)  # [n_spots_t5A]
        scoreA = get_score(distancesA, 2)  # float

        tile_5B_yx_shifted = tile_5B_yx + shift_yx   # [n_spots_t5B x 2]
        distances = find_neighbour_distances(tile_5B_yx_shifted, 
                                             tile_4B_yx)  # [n_spots_t5B]
        scoreB = get_score(distancesB, 2)  # float
        score = scoreA + scoreB
    best_shift_yx is the shift_yx with the best score
    ```
    
Once the best $yx$ shift is found, the shift in $z$ is found by using the full *3D* point clouds again
but doing just an exhaustive search in $z$. 

Before this is done, it is important that the $z$ coordinate of the point clouds is in the same unit 
as the $yx$ coordinate so distances are computed correctly.
The conversion from $z$ pixel units to $yx$ pixel units is achieved by multiplying 
the $z$ coordinate by `nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy`. The $z$ shifts in the 
exhaustive search must also be put into $yx$ pixel units.

??? example

    If the previous example found the best $yx$ shift to be `best_shift_yx`, 
    the pseudocode below is what [follows](../code/stitch/shift.md#iss.stitch.shift.get_best_shift_3d) 
    this to find the best $yxz$ shift.

    ```
    # tile_5_yxz: [n_spots_t5 x 3]
    # tile_4_yxz: [n_spots_t4 x 3]
    # exhaustive_search_z: [n_shifts_z]
    # best_shift_yx: [2]

    shift_yxz = [0, 0, 0]
    The yx shift is constant in this search, always set to the best 
    yx shift we found in the 2D search.
    shift_yxz[0] = best_shift_yx[0]
    shift_yxz[1] = best_shift_yx[1]
    for shift_z in exhaustive_search_z:
        shift_yxz[2] = shift_z
        tile_5_yxz_shifted = tile_5_yxz + shift_yxz   # [n_spots_t5 x 3]
        distances = find_neighbour_distances(tile_5_yxz_shifted, 
                                             tile_4_yxz)  # [n_spots_t5]
        score = get_score(distances, 2)  # float
    best_shift is the shift with the best score
    ```

### Score Threshold

### Widening range

### Refined search

### Updating range

## Global coordinates

## Saving stitched images
    
## View stitched point clouds
