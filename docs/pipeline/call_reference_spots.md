# Call Reference Spots
The [*call reference spots* step of the pipeline](../code/pipeline/call_reference_spots.md) uses the 
$n_{rounds} \times n_{channels}$  `color` obtained for each reference spot 
(detected on the reference round/reference channel ($r_{ref}$/$c_{ref}$) in the 
[*get reference spots*](get_reference_spots.md) step of the pipeline to compute the `bleed_matrix` accounting
for crosstalk between color channels. We then compute the `gene_efficiency` which allows for varying round
strengths for each gene, before assigning each reference spot to a gene.

These gene assignments are saved in the [`ref_spots`](../notebook_comments.md#ref_spots) *NotebookPage* 
while the bleed_matrix and expected bled_code for each gene are saved in the 
[`call_spots`](../notebook_comments.md#call_spots) *NotebookPage*.

!!! note "Note: `config` in this section means `config['call_spots']`"

## Gene Assignment
### Color Normalisation
We assign a spot $s$ to a gene $g$, based on its $n_{rounds} \times n_{channels}$ `color`, $\pmb{\acute{\zeta}_s}$, 
indicating the intensity in each round and channel.

However, we first need to equalize color channels, so that no one color channel dominates the others when 
it comes to gene assignment. The normalised `spot_colors`, $\pmb{\zeta}$, are obtained by dividing the saved 
`spot_colors`, $\pmb{\acute{\zeta}}$, by a $n_{rounds} \times n_{channels}$ 
normalisation factor, `nb.call_spots.color_norm_factor`.

??? question "Why do we need to normalise color channels?"

    The gene assignment of the spot below indicates why we need to normalise color channels.
    With no normalisation, the gene assignment is overly influenced by channel 4 which 
    is one of the most intense channels (see `color_norm_factor` in example box below). Thus it matches 
    to *Reln* which appears in channel 4 in rounds 0, 1, 5 and 6. It also doesn't care at all about 
    channel 2 because it is the weakest channel and *Reln* does not appear in channel 2 in any rounds.

    With normalisation, the most obvious effect is channel 2 gets boosted and now has an influence.
    Given that *Id2* appears in channel 2 in rounds 1, 3, 5 and 6, if channel 2 was never considered,
    it would always make the score to *Id2* low. When we include channel 2 though, we get a high score
    even though we are not matching the round 0, channel 4 intensity which contributed to the *Reln* assignment
    without normalisation.
    
    === "No Normalisation"
        ![image](../images/pipeline/call_spots/spot_color_no_norm.png){width="800"}

    === "With Normalisation"
        ![image](../images/pipeline/call_spots/spot_color_with_norm.png){width="800"}

This is [obtained](../code/call_spots/base.md#iss.call_spots.base.color_normalisation) 
using the parameters `config['color_norm_intensities']` and 
`config['color_norm_probs']` such that for each round, $r$, and channel $c$, 
the probability of $\zeta_{rc}$ being larger than 
`config['color_norm_intensities'][i]` is less than `config['color_norm_probs'][i]` for each $i$.

The probabilities come from the [histograms](extract.md#hist_counts) produced in the *extract and filter* step i.e.
if `config['color_norm_intensities'] = 0.5` and `config['color_norm_probs'] = 0.01`, then 
in each round and channel, only 1% of pixels on the mid z-plane across all tiles would have $\zeta_{rc} > 0.5$.

If `config[bleed_matrix_method] = 'single'`, then we combine all rounds for each channel so that 
`nb.call_spots.color_norm_factor[r, c]` is the same for all $r$ of a particular $c$. 

??? example

    With `config['color_norm_intensities'] = 0.5, 1, 5` and `config['color_norm_probs'] = 0.01, 5e-4, 1e-5`
    and the [histograms](extract.md#hist_counts) shown in the *extract and filter* step, the two
    methods of `config[bleed_matrix_method]` produce the following `color_norm_factor`:
    
    === "Single"
        ![image](../images/pipeline/call_spots/color_norm_factor.png){width="500"}

    === "Separate"
        ![image](../images/pipeline/call_spots/color_norm_factor_sep.png){width="500"}

    The [normalised histogram](extract.md#hist_counts) shown was normalised using the *Single* `color_norm_factor`
    and you can see that for each round and channel there is a similar area under the curve
    (probability) beyond $\zeta_{rc}=0.5$, as expected from `config['color_norm_intensities']`.

### Background
After we have the normalised spot colors, $\pmb{\zeta}$, we 
[remove](../code/call_spots/background.md#iss.call_spots.background.fit_background) some background *genes* from them. 
There is one background *gene* for each channel, $\pmb{B}_C$. The background *gene* for channel $C$ is defined by:
$\pmb{B}_{C_{rc}}$ if $c = C$ and 0 otherwise i.e. it is just a strip in channel $C$:

![image](../images/pipeline/call_spots/background/channel1_background.png){width="300"}

It is also normalised to have an L2 norm of 1. These are saved as `nb.call_spots.background_codes`.

??? question "Why do we need to fit background *genes*?"

    We fit the background *genes* because it is fairly common for $\pmb{\zeta}_s$ to have high intensity 
    across all rounds of a channel as shown for the example [`view_codes`](../view_results.md#c-view_codes) 
    plot below:
    
    === "No Background Removal"
        ![image](../images/pipeline/call_spots/background/thsd7a_spot_no_background_remove.png){width="500"}

    === "With Background Removal"
        ![image](../images/pipeline/call_spots/background/thsd7a_spot_background_remove.png){width="500"}

    Now no gene in the codebook looks that much like a background gene but if the background genes have not
    been fit, as with the first image above, spots like this will match to the gene which has the most 
    rounds in the relavent channel/s. Here, *Thsd7a* has intensity in channel 2 in all rounds apart from round 2.
    This problem will be exacerbated in the omp step, because at each iteration of the omp algorithm, 
    it will just try and fit more and more genes to explain the intense channel.

    If we do remove background though, as with the second image, the gene assignment will be based 
    on the other channels where not all rounds were intense. In this case, we get a match to *Sst* due
    to round 2 and 3 in channel 0.

    If we look at [`histogram_score`](#histogram_score) for *Thsd7a*, we see that wihtout background removal
    (purple), the peak in score is significantly larger:

    ![image](../images/pipeline/call_spots/background/thsd7a_hist.png){width="500"}

    This indicates that without background removal, we would end up with a lot more spots assigned to
    genes like *Thsd7a* which have high intensity in most rounds of a single channel.
    

The coefficient, $\mu_{sC}$, for the channel $C$ background *gene*, $\pmb{B}_C$, fit to the spot $s$ color, 
$\pmb{\zeta}_s$ is:

$$
\mu_{sC} = \frac{\sum_rw^2_{rC}\zeta_{s_{rC}}B_{C_{rC}}}{\sum_rw^2_{rC}B^2_{C_{rC}}}; 
w_{rC} = \frac{1}{|\zeta_{s_{rC}}| + \lambda_b}
$$

Where, $\lambda_b$ is `config['background_weight_shift']` and the sum is over all rounds used.

??? note "Value of $\lambda_b$"

    If `config['background_weight_shift']` is left blank, it is set to the median of the [`intensity`](#intensity) 
    computed from the absolute colors of all pixels in the middle z-plane 
    (`nb.call_spots.norm_shift_tile`) of the central tile (`nb.call_spots.norm_shift_z`).

    This is because it gives an estimate of what $|\zeta_{s_{rC}}|$ would be for an average non-spot pixel.
    If we set $\lambda_b$ to be equal to this, it is saying that if a spot had a round and channel with very
    low intensity, then that round and channel would have as much influence on the final coefficient
    as an average pixel. 

    If $\lambda_b = 0$, then $w_{rc}$ would go to $\infty$ for low intensity rounds and channels,
    so the value of $\lambda_b$ chosen also provides an upper bound onthe contribution of low intensity 
    rounds and channels to the final coefficient.

If the weighting, $w$, was a constant across rounds and channels ($\lambda_b = \infty$), this would just be 
the least squares solution. After we have found the coefficients, we remove the background contribution
from the spot colors to give $\pmb{\zeta}_{{s0}}$ which we use from now on.

$$
\zeta_{{s0}_{rC}} = \zeta_{{s}_{rC}} - \mu_{sC}B_{C_{rC}}
$$


??? question "Why do we need a weighting?"
    
    We find $\mu_{sC}$ via weighted least squares, because it limits the influence of outliers. 
    The example below shows that with $\lambda_b = \infty$ (normal least squares), really tries to remove
    the outlier in round 4, channel 0. The result of this is that all the other rounds of channel 0 become 
    very negative.
    
    === "$\pmb{\zeta}_s$"
        ![image](../images/pipeline/call_spots/background/view_codes.png){width="600"}
    === "$\pmb{\zeta}_{s0} (\lambda_b = 0.08)$"
        ![image](../images/pipeline/call_spots/background/view_codes_0_08_weight.png){width="600"}
    === "$\pmb{\zeta}_{s0} (\lambda_b = \infty)$"
        ![image](../images/pipeline/call_spots/background/view_codes_infinite_weight.png){width="600"}
    
    This spot should be *Slc6a1* as shown from the $\lambda_b = 0.08$ image but *Slc6a1* is expected to be 
    in 4 of the 6 rounds that were set to negative by the background fitting with $\lambda_b = \infty$.
    Thus, this spot can no longer be assigned to the correct gene after $\lambda_b = \infty$ background fitting.
    
    So basically, the job of the background fitting is to remove intensity in a particular channel if 
    that channel is intense across all rounds. This is because there are no actual genes which can explain this.
    We do not want it to tone down outlier rounds and channels because outliers are usually due to actual genes.



#### [`view_background`](../code/plot/call_spots.md#view_background)
The background coefficient calculation can be visualised by using the 
[`view_background`](../code/plot/call_spots.md#view_background) function:

![image](../images/pipeline/call_spots/background/thsd7a_background_fit.png){width="800"}

For each plot, each row corresponds to a different background *gene* coefficient calculation i.e. a different 
channel. There is no overlap between the background codes hence we can view all the calculations at the same time.

In the bottom row, round $r$, channel $c$ in the *Weighted Dot Product* plot refers to the value
of $\frac{w^2_{rc}\zeta_{s_{rc}}B_{C_{rc}}}{\sum_rw^2_{rC}B^2_{C_{rC}}}$. 
In the top row, the *Dot Product* plot is the same as the *Weighted Dot Product* plot except $\lambda_b = \infty$.

The bottom row *Weighted Coef* plot thus shows the coefficient computed for the current value of $\lambda_b$ 
(0.08 here, but can be specified with in the textbox). The top row *Coef* plot shows the coefficient that would
be computed with $\lambda_b = \infty$.

The main difference between the two in this case is that the channel 0 coefficient is much larger for the
$\lambda_b = \infty$ case. This is because the *Weight Squared*, $\Omega^2_{s_{rc}}$ term acts to increase the
contribution of the weak round 1, channel 0 and decrease the contribution of the strong rounds: 0, 2, 3 and 6.

### Bleed Matrix
Crosstalk can occur between color channels. Some crosstalk may occur due to optical bleedthrough; 
additional crosstalk can occur due to chemical cross-reactivity of probes. The precise degree of crosstalk does not
seem to vary much between sequencing rounds.
It is therefore possible to largely compensate for this crosstalk by learning the precise amount of crosstalk 
between each pair of color channels.

To [estimate the crosstalk](../code/call_spots/bleed_matrix.md#iss.call_spots.bleed_matrix.get_bleed_matrix), 
we use the spot colors, $\pmb{\zeta}_{s0}$, of all well isolated spots. We 
reshape these, so we have a set of $n_{isolated} \times n_{rounds}$ vectors, each of dimension 
$n_{channels}$, $\pmb{v}_i$ ($v_{0_c} = \zeta_{{s=0,0}_{r=0,c}}$).
Only [well-isolated](find_spots.md#isolated-spots) spots are used to ensure that crosstalk estimation is 
not affected by spatial overlap of spots corresponding to different genes.

Crosstalk is then estimated by running a 
[scaled k-means](../code/call_spots/bleed_matrix.md#iss.call_spots.bleed_matrix.scaled_k_means) 
algorithm on these vectors, which finds a set of $n_{dye}$ vectors, $\pmb{c}_d$, such that the error function:

$$
\sum_i\min_{\lambda_i, d(i)}|\pmb{v}_i - \lambda_i\pmb{c}_{d(i)}|^2
$$

is minimized. In other words, it finds the $n_{dyes}$ intensity vectors, $\pmb{c}_d$, each of dimension
$n_{channels}$, such that the spot color of each well isolated spot on every round is close to a scaled version of 
one of them. The $n_{dyes} \times n_{channels}$
array of dye vectors is termed the `bleed matrix` and is saved as `nb.call_spots.bleed_matrix`.
The [`view_bleed_matrix`](../view_results.md#b-view_bleed_matrix) function can be used to show it:


=== "Single Bleed Matrix"
    ![image](../images/pipeline/call_spots/bleed_matrix/single.png){width="800"}
=== "Separate Bleed Matrix For Each Round"
    ![image](../images/pipeline/call_spots/bleed_matrix/separate.png){width="800"}


As shown in the second plot, if `config['bleed_matrix_method'] = 'separate'`, we compute a different 
`bleed_matrix` for each round - i.e. we loosen the assumption that crosstalk does not vary between sequencing rounds.

#### Initial Bleed Matrix
To estimate the dye intensity vectors, $\pmb{c}_d$, the 
[`scaled_k_means`](../code/call_spots/bleed_matrix.md#iss.call_spots.bleed_matrix.scaled_k_means) algorithm
needs to know the number of dyes and a starting guess for what each dye vector looks like.

This is specified in the [`basic_info`](../config.md#basic_info) section of the configuration file as explained 
[here](../config_setup.md#specifying-dyes).

#### Scaled K Means
The pseudocode for the [`scaled_k_means`](../code/call_spots/bleed_matrix.md#iss.call_spots.bleed_matrix.scaled_k_means)
algorithm to obtain the dye intensity vectors, $\pmb{c}_d$, is given below:

```
v: Single round intensity vectors of well isolated spots 
   [n_vectors x n_channels]
c: Initial Guess of Bleed Matrix
   [n_dyes x n_channels]
dye_ind_old: [n_vectors] array of zeros.

v_norm = v normalised so each vector has an L2 norm of 1.
Normalise c so that each dye vector has an L2 norm of 1.

i = 0
while i < n_iter:
    score = dot product between each vector in v_norm and each
        dye in c. [n_vectors x n_dyes]
    top_score = highest score across dyes for each vector
        in v_norm. [n_vectors].
    dye_ind = dye corresponding to top_score for each vector
        in v_norm. [n_vectors].
        
    if dye_ind == dye_ind_old:
        Stop iteration because we have reached convergence
        i.e. i = n_iter
    dye_ind_old = dye_ind
    
    for d in range(n_dyes):
        v_use = all vectors in v with dye_ind = d and 
            top_score > score_thresh.
            Use un-normalised as to avoid overweighting weak points.
            [n_use x n_channels]
        if n_use < min_size:
            c[d] = 0
        else:
            Update c[d] to be top svd component of v_use i.e.
                v_matrix = v_use.transpose() @ v_use / n_use
                    [n_channels x n_channels]
                c[d] = np.linalg.svd(v_matrix)[0][:, 0]
                    [n_channels]
        
    i = i + 1    
return c
```
There are a few parameters in the config file which are used:

* `bleed_matrix_n_iter`: This is `n_iter` in the above code.
* `bleed_matrix_min_cluster_size`: This is `min_size` in the above code.
* `bleed_matrix_score_thresh`: This is `score_thresh` in the above code.
* `bleed_matrix_anneal`: If this is `True`, the `scaled_k_means` algorithm will be run twice.
    The second time will use $\pmb{c}_d$ returned from the first run as the starting point,
    and it will have a different `score_thresh` for each dye. `score_thresh` for dye $d$ 
    will be equal to the median of `top_score[dye_ind == d]` in the last iteration of the first run.
    The idea is that for the second run, we only use vectors which have a large score,
    to get a more accurate estimate.

#### [`view_scaled_k_means`](../code/plot/call_spots.md#scaled-k-means)
The [`scaled_k_means`](../code/call_spots/bleed_matrix.md#iss.call_spots.bleed_matrix.scaled_k_means) algorithm 
can be visualised using the [`view_scaled_k_means`](../code/plot/call_spots.md#scaled-k-means) function:

![image](../images/pipeline/call_spots/bleed_matrix/scaled_k_means.png){width="800"}

In each column, the top row, boxplot $d$, is for `top_score[dye_ind == d]` (only showing scores above
`score_thresh`, which is 0 for the first two plots) with the dye vectors, $\pmb{c}_d$, indicated in the second
row. The number of vectors assigned to each dye is indicated by the number within each boxplot.

The first column is for the first iteration i.e. with the initial guess of the bleed matrix.
The second column is after the first `scaled_k_means` algorithm has finished.
The third column is after the second `scaled_k_means` algorithm has finished (only shown if `bleed_matrix_anneal=True`).
The bottom whisked of the boxplots in the third column indicate the `score_thresh` used for each dye.

This is useful for debugging the `bleed_matrix` computation, as you want the boxplots to show high scores and for those
scores to increase from left to right as the algorithm is run.

### Gene Bled Codes
Once the `bleed_matrix` has been computed, the expected code for each gene can be 
[obtained](../code/call_spots/base.md#iss.call_spots.base.get_bled_codes).

Each gene appears with a single dye in each imaging round as indicated by the [codebook](../config_setup.md#code_book)
and saved as `nb.call_spots.gene_codes`.
`bled_code[g, r, c]` for gene $g$ in round $r$, channel $c$ is then given by `bleed_matrix[r, c, gene_codes[r]]`.

Each `bled_code` is also normalised to have an L2 norm of 1. They are saved as `nb.call_spots.bled_codes`.

??? example

    Using the *Single Bleed Matrix* shown as an example [earlier](#bleed-matrix), the bled_code
    for *Kcnk2* with `gene_code = 6304152` is:

    ![image](../images/pipeline/call_spots/bleed_matrix/bled_code.png){width="800"}

### Dot Product Score
To assign spot $s$, with spot color (post background), $\pmb{\zeta}_{{si}}$, to a gene, we compute a dot product
score, $\Delta_{sig}$ to each gene, $g$, with `bled_code` $\pmb{b}_g$. This is defined to be:

$$
\Delta_{sig} = \sum_{r=0}^{n_r-1}\sum_{c=0}^{n_c-1}\omega^2_{{si}_{rc}}\tilde{\zeta}_{{si}_{rc}}b_{g_{rc}}
$$

Where:

$$
\tilde{\zeta}_{{si}_{rc}} = \frac{\zeta_{{si}_{rc}}}
{\sqrt{\sum_{\mathscr{r}=0}^{n_r-1}\sum_{\mathscr{c}=0}^{n_c-1}\zeta^2_{{si}_{\mathscr{rc}}}} + \lambda_d}
$$

$$
\sigma^2_{{si}_{rc}} = \beta^2 + \alpha\sum_g\mu^2_{sig}b^2_{g_{rc}}
$$

$$
\omega^2_{{si}_{rc}} = n_rn_c\frac{\sigma^{-2}_{{si}_{rc}}}
{\sum_{\mathscr{r}=0}^{n_r-1}\sum_{\mathscr{c}=0}^{n_c-1}\sigma^{-2}_{{si}_{\mathscr{rc}}}}
$$

* $n_{r}$ is the number of rounds.
* $n_{c}$ is the number of channels.
* $\lambda_d$ is `config['dp_norm_shift'] * sqrt(n_rounds)` (typical `config['dp_norm_shift']` is 0.1).
* $\alpha$ is `config['alpha']` (*default is 120*).
* $\beta$ is `config['beta']` (*default is 1*).
* The sum over genes, $\sum_g$, is over all real and background genes i.e. $\sum_{g=0}^{n_g+n_c-1}$
* $i$ refers to the number of actual genes fit prior to this iteration of *OMP*. Here, because we are
only fitting one gene, $i=0$, meaning only background has been fit.

So if $\lambda_d = 0$ and $\pmb{\omega}^2_{si}$ was 1 for each round and channel (achieved through $\alpha=0$), then 
this would just be the normal dot product between two vectors with L2 norm of one. 
The min value is 0 and max value is 1. 

The purpose of the weighting, $\pmb{\omega}^2_{{si}}$, is to decrease the contribution of rounds/channels 
which already have a gene in. It is really more relevant to the *OMP* algorithm. 
It can be turned off in this part of the pipeline by setting
`config['alpha'] = 0`. The $n_rn_c$ term at the start of the $\omega^2_{{si}_{rc}}$ equation is a normalisation
term such that the max possible value of $\Delta_{sig}$ is approximately 1 (it can be more though).

??? note "Value of $\lambda_d$"

    If in the $\Delta_{sig}$ equation, we used $\pmb{\zeta}_{{si}}$ instead of $\pmb{\tilde{\zeta}}_{{si}}$,
    then the max value would no longer have an upper limit and a high score could be achieved by having 
    large values of $\pmb{\zeta}_{{si}}$ in some rounds and channels as well as by having $\pmb{\zeta}_{{si}}$
    being similar to $\pmb{b}_g$. 

    We use the [intensity](#intensity) value to indicate the strength of the spot, so for $\Delta_{sig}$, we 
    really just want a variable which indicates how alike the spot color is to the gene, indepenent of
    strength. Hence, we use $\pmb{\tilde{\zeta}}_{{si}}$.

    If $\lambda_d = 0$, it would mean that even background pixels with very small intensity could get a high score.
    So, we use a non-zero value of $\lambda_d$ to prevent very weak spots getting a large $\Delta_{sig}$.

    If `config['dp_norm_shift']` is not specified, it is set to the median of the L2 norm in a single round
    computed from the colors of all pixels in the middle z-plane 
    (`nb.call_spots.norm_shift_tile`) of the central tile (`nb.call_spots.norm_shift_z`).
    
    The idea behind this is that, the L2 norm of an average background pixel would be 
    `config['dp_norm_shift'] * sqrt(n_rounds)`. So if $\lambda_d$ was set to this, it is giving 
    a penalty to any spot which is less intense than the average background pixel. This is 
    desirable since any pixels of such a low intensity are unlikely to be spots.

The spot $s$, is assigned to the gene $g$, for which $\Delta_{sig}$ is the largest.

#### [`view_score`](../code/plot/call_spots.md#view_score)
How the various parameters in the dot product score calculation affect the final value can be investigated,
for a single spot, through the function [`view_score`](../code/plot/call_spots.md#view_score):


=== "`view_score`"
    ![image](../images/pipeline/call_spots/view_score.png){width="800"}
=== "`view_weight`"
    ![image](../images/pipeline/call_spots/view_weight.png){width="800"}

* The top left plot shows the spot color prior to background removal.
* The bottom left plots shows the spot color after background removal.
* The top right plot shows the dot product score obtained without weighting ($\alpha=0$).
* The bottom right plot shows the actual score obtained with the current weighting parameters.
* To get to the gene with the highest dot product score for the current iteration, you can enter an impossible value
in the *Gene* textbox. As well as typing the index of the gene, you can also type in the gene name to 
look at the calculation for a specific gene.
* Clicking on the *Weight Squared* plot shows the [`view_weight`](../code/plot/call_spots.md#view_weight) 
plot indicating how it is calculated (second image above).

Looking at the `view_weight` image and the far right plots of the `view_score` image, 
we see that the effect of the weighting is to down-weight color channel 0 because this is where the background
coefficient is the largest. The channels with a smaller background coefficient (1, 5 and 6) then have 
a weighting greater than 1. Thus, the weighted score is greater than the non-weighted one because
channel 6, where this spot is particularly strong has a greater contribution.
I.e. because the intensity in channel 6 cannot be explained by background genes, but it can be explained by Plp1, 
we boost the score. For most spots, the background coefficients are very small and so the weighting has little effect.

Using the [`histogram_score`](#histogram_score) function, we see that the effect of weighting (blue) is to 
add a tail of scores greater than 1 for spots where an increased contribution is given to the rounds/channels
where they are most intense. The mode score does not change though:

![image](../images/pipeline/call_spots/hist_weight.png){width="800"}

### Gene Efficiency



## Intensity

## Diagnostics
### [`histogram_score`](../code/plot/omp.md#histogram_score)