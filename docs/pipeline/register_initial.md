# Register Initial
The [*register initial* step of the pipeline](../code/pipeline/register_initial.md) uses the point clouds
added to the *Notebook* during the [`find_spots`](find_spots.md) step
to find the shift between the reference round and each imaging round for every tile.
The $yxz$ shift to tile $t$, round $r$ is saved as `nb.register_initial.shift[t, r]`. This shift
is then used as a starting point for finding the affine transforms in the 
[*register*](register.md) step of the pipeline 
to all channels of tile $t$, round $r$.

The [`register_initial`](../notebook_comments.md#register_initial) *NotebookPage* is added to the 
*Notebook* after this stage is finished.

## Shift
The channel used for finding the shifts is specified by $c_{shift}$ = `config['register_initial']['shift_channel']`.
If it is left blank, it will be set to $c_{ref}$ (`nb.basic_info.ref_channel`). This channel
should be one with lots of spots and if an [error](#error-too-many-bad-shifts) 
is hit, it may be worth re-running with a different value
of this parameter. 

So, for tile $t$, round $r$, we find the shift between $r_{ref}$/$c_{ref}$ and $r$/$c_{shift}$.

The [function](../code/stitch/shift.md#iss.stitch.shift.compute_shift) to compute these shifts is exactly the 
same as the one used in the [stitch](stitch.md#shift) section of the pipeline and the parameters in the 
[*register initial*](../config.md#register_initial) section of the config file do the same thing as the 
corresponding parameters in the [*stitch*](../config.md#stitch) section. A few details are different though,
as explained below.

### Initial range
The difference to the [*stitch*](stitch.md#initial-range) case is that `config['register_initial']['shift_min']` and 
`config['register_initial']['shift_max']` are always used. We expect the shift between rounds to be quite 
small hence the default values which perform an exhaustive search centered on 0 in each direction
with a range of 200 in $y$ and $x$ and a range of 6 in $z$.

### Updating initial range
We assume that the shifts to a given round will be approximately the same for all tiles. So, after we have found 
at least 3 shifts to a round which have `score > score_thresh`, we update our initial exhaustive search range
to save time for future tiles. See the [example](stitch.md#updating-initial-range) 
in the *stitch* section for how the update is performed.

### Amend low score shifts
This is very similar to the [*stitch*](stitch.md#amend-low-score-shifts) case, 
but the names of the variables saved to the *Notebook* are slightly different:

After the shifts to all rounds for all tiles have been found, the ones with `score < score_thresh` are 
amended.

If for round $r$, tile $t$, the best shift found had a `score < score_thresh`, 
the shift and score are saved in the notebook in `nb.register_initial.shift_outlier[t, r]` and 
`nb.register_initial.shift_score_outlier` respectively.

The shift is then re-computed using a new initial exhaustive search range 
(saved as `nb.register_initial.final_shift_search`). This range is computed using the 
[`update_shifts`](#updating-initial-range) function to centre 
it on all the shifts found to round $r$ for which `score > score_thresh`.
For this re-computation, no [widening](stitch.md#widening-range) is allowed either. The idea behind this 
is that it will force the shift to be within the range we expect based on the successful shifts. 
I.e. a shift with a slightly lower `score` but with a shift
more similar to the successful shifts is probably more reliable than a shift with 
a slightly higher `score` but with a shift significantly different from the successful ones.

The new shift and score will be saved in `nb.register_initial.shift[t, r]` and 
`nb.register_initial.shift_score[t, r]` respectively.

## Error - too many bad shifts
After the `register_initial` *NotebookPage* has been 
[added](../code/pipeline/run.md#iss.pipeline.run.run_register) to the *Notebook*, 
[`check_shifts_register`](../code/stitch/check_shifts.md#iss.stitch.check_shifts.check_shifts_register) will be run.

This will produce a warning for any shift found with `score < score_thresh`.

An error will be raised if the fraction of shifts with `score < score_thresh` exceeds 
`config['register_initial']['n_shifts_error_fraction']`.

If this error does occur, it is probably worth looking at the [debugging plots](#debugging) to see if the shifts found
looks good enough to use as a starting point for the [iterative closest point algorithm](register.md)
or if it should be re-run with different configuration
file parameters (e.g. different `config['register_initial']['shift_channel']` corresponding to a channel
with more spots, smaller `config['register_initial']['shift_step']` or larger 
`config['register_initial']['shift_max_range']`). 

## Debugging