# Generate random data for integration testing in coppafish
#
# Synthetic data is added in a modular way to the RoboMinnie class.
#
# Currently, the biggest limitation is that this only works for one tile.  To
# get it to work with multiple tiles, we would have to specify a geometry and
# then generate the background noise on the minimal rectangle containing that
# space.  This would include the overlap for each.  Then, we would sample spot
# positions within these tile boundaries, and if it falls into an overlap, then
# put it on both tiles.
#
# Originally by Max Shinn, August 2023
# Refactored and expanded by Paul Shuker, September 2023
import os
import numpy as np
import scipy.stats
import pandas
import dask.array
from typing import Tuple
import warnings
import json
import inspect
import time
import pickle
from tqdm import tqdm, trange
import bz2
import napari

# Typing imports
from typing import Dict, List, Any
import numpy.typing as npt
from numpy.random import Generator


DEFAULT_IMAGE_DTYPE = float
DEFAULT_INSTANCE_FILENAME = 'robominnie.pkl'
USE_INSTANCE_GENE_CODES = None


def _funcname():
    return str(inspect.stack()[1][3])


def _compare_spots(
    spot_positions_yxz : npt.NDArray, spot_gene_indices : npt.NDArray, true_spot_positions_yxz : npt.NDArray, \
    true_spot_gene_identities : npt.NDArray, location_threshold_squared : float, codes : Dict[str,str],
    description : str,
    ) -> Tuple[int,int,int,int]:
    """
    Compare two collections of spots (one is the ground truth) based on their positions and gene identities.

    args:
        spot_positions_yxz ((n_spots x 3) ndarray): The assigned spot positions
        spot_gene_indices ((n_spots) ndarray): The indices for the gene identities assigned to each spot. The \
            genes are assumed to be in the order that they are found in the genes parameter
        true_spot_positions_yxz (ndarray): The ground truth spot positions
        true_spot_gene_identities ((n_true_spots) ndarray): Array of every ground truth gene name, given as a \
            `str`
        location_threshold_squared (float): The square of the maximum distance two spots can be apart to be \
            paired.
        codes (dict of str: str): Each code name as a key is mapped to a unique code, both stored as `str`
        description (str, optional): Description of progress bar for printing. Default: empty

    Returns:
        Tuple[int,int,int,int]: _description_
    """
    true_positives  = 0
    wrong_positives = 0
    false_positives = 0
    false_negatives = 0

    spot_count = spot_positions_yxz.shape[0]
    true_spot_count = true_spot_positions_yxz.shape[0]
    # Stores the indices of every true spot index that has been paired to a spot already
    true_spots_paired = np.empty(0, dtype=int)
    for s in trange(spot_count, ascii=True, desc=description, unit='spots'):
        x = spot_positions_yxz[s,1]
        y = spot_positions_yxz[s,0]
        z = spot_positions_yxz[s,2]
        position_s = np.repeat([[y, x, z]], true_spot_count, axis=0)
        # Subtract the spot position along all true spots
        position_delta = np.subtract(true_spot_positions_yxz, position_s)
        position_delta_squared = \
            position_delta[:,0] * position_delta[:,0] + \
            position_delta[:,1] * position_delta[:,1] + \
            position_delta[:,2] * position_delta[:,2]
        
        # Find true spots close enough and closest to the spot, stored as a boolean array
        matches = np.logical_and(position_delta_squared <= location_threshold_squared, \
            position_delta_squared == np.min(position_delta_squared))
        
        # True spot indices
        matches_indices = np.where(matches)
        delete_indices = []
        if np.sum(matches) > 0:
            # Ignore true spots close enough to the spot if already paired to a previous spot
            for i in range(len(matches_indices)):
                if matches_indices[i] in true_spots_paired:
                    delete_indices.append(i)
                    matches[matches_indices[i]] = False
        delete_indices = np.array(delete_indices, dtype=int)
        if delete_indices.size > 0:
            matches_indices = np.delete(matches_indices, delete_indices)
        matches_count = np.sum(matches)

        if matches_count == 0:
            # This spot is considered a false positive because there are no true spots close enough to it that 
            # have not already been paired
            false_positives += 1
            continue
        if matches_count == 1:
            # Found a single true spot matching the spot, see if they are the same gene (true positive)

            # Get the spot gene name from the gene number. For some reason coppafish adds a new gene called 
            # Bcl11b, hence the -1
            spot_gene_name = \
                str(list(codes.keys())[spot_gene_indices[s]])
            # Actual true spot gene name as a string
            true_gene_name = str(true_spot_gene_identities[matches][0])
            matching_gene = spot_gene_name == true_gene_name
            true_positives  += matching_gene
            wrong_positives += not matching_gene
            true_spots_paired = np.append(true_spots_paired, matches_indices[0])
            continue
        
        # Logic for dealing with multiple, equidistant true spots near the spot
        for match_index in matches_indices:
            spot_gene_name = \
                str(list(codes.keys())[spot_gene_indices[s]])
            # Actual true spot gene names as strings
            true_gene_name = str(true_spot_gene_identities[matches[match_index][0]])
            matching_gene = spot_gene_name == true_gene_name
            if matching_gene:
                true_positives += 1
                true_spots_paired = np.append(true_spots_paired, match_index)
                continue
        # If reaching here, all close true spots are not the spot gene
        # Assign the first true spot in the array as the pair and label it a wrong_positive
        true_spots_paired = np.append(true_spots_paired, matches_indices[0])
        wrong_positives += 1
        continue
    # False negatives are any true spots that have not been paired to a spot
    false_negatives = true_spot_count - true_spots_paired.size
    return (true_positives, wrong_positives, false_positives, false_negatives)


class RoboMinnie:
    """
    RoboMinnie
    ==========
    Fast coppafish integration suite
    
    Provides:
    ---------
    1. Single tile, modular, customisable synthetic data generation for coppafish
    2. Coppafish raw .npy file generation for pipeline running
    3. Coppafish scoring using ground-truth spot data

    Usage:
    ------
    Create new RoboMinnie instance for each integration test. Call functions for data generation (see \
        ``RoboMinnie.py`` functions for options). Call ``Save_Coppafish`` then ``Run_Coppafish``. Use \
        ``Compare_Spots_OMP`` to evaluate OMP results.
    """
    #TODO: Multi-tile support
    #TODO: DAPI support
    def __init__(self, n_channels : int = 7, n_tiles : int = 1, n_rounds : int = 7, n_planes : int = 4, 
        n_yx : Tuple[int, int] = (2048, 2048), include_anchor : bool = True, include_presequence : bool = True, 
        include_dapi : bool = True, anchor_channel : int = 1, dapi_channel : int = 0, tiles_width : int = None, \
        image_dtype : Any = None, seed : int = 0) -> None:
        """
        Create a new RoboMinnie instance. Used to manipulate, create and save synthetic data in a modular,
        customisable way. Synthetic data is saved as raw .npy files and includes everything needed for coppafish 
        to run a full pipeline.

        args:
            n_channels (int, optional): Number of channels. Default: 7.
            n_tiles (int, optional): Number of tiles. Default: 1.
            n_rounds (int, optional): Number of rounds. Not including anchor or pre-sequencing. Default: 7.
            n_planes (int, optional): Number of z planes. Default: 4.
            n_yx (Tuple[int, int], optional): Number of pixels for each tile in the y and x directions \
                respectively. Default: (2048, 2048).
            include_anchor (bool, optional): Whether to include the anchor round. Default:  true.
            include_presequence (bool, optional): Whether to include the pre-sequence round. Default: true.
            include_dapi (bool, optional): Whether to include a DAPI image. Uses a single channel, stains the \
                cell nuclei so they light up to find cell locations. Default: true.
            anchor_channel (int, optional): The anchor channel. Default: 1.
            dapi_channel (int, optional): The DAPI image channel. It is saved as part of the anchor raw file \
                Default: 0.
            tiles_width (int, optional): Number of tiles aligned along the x axis together. Default: \
                `floor(sqrt(n_tiles))`.
            image_dtype (any, optional): Datatype of images. Default: float
            seed (int, optional): Seed used throughout the generation of random data, specify integer value for \
                reproducible output. If None, seed is randomly picked. Default: 0.
        """
        self.n_channels = n_channels
        self.n_tiles = n_tiles
        self.n_rounds = n_rounds
        self.n_planes = n_planes
        self.n_yx = n_yx
        self.n_spots = 0
        self.bleed_matrix = None
        self.true_spot_positions_pixels = np.zeros((0, 3)) # Has shape n_spots x 3 (x,y,z)
        self.true_spot_identities = np.zeros((0), dtype=str) # Has shape n_spots, saves every spots gene
        self.include_anchor = include_anchor
        self.include_presequence = include_presequence
        self.include_dapi = include_dapi
        self.anchor_channel = anchor_channel
        self.dapi_channel = dapi_channel
        if image_dtype == None:
            self.image_dtype = DEFAULT_IMAGE_DTYPE
        else:
            self.image_dtype = image_dtype
        self.seed = seed
        self.instructions = [] # Keep track of the functions called inside RoboMinnie, in order
        self.instructions.append(_funcname())
        assert self.n_channels > 0, 'Require at least one channel'
        assert self.n_tiles > 0, 'Require at least one tile'
        assert self.n_rounds > 0, 'Require at least one round'
        assert self.n_planes > 0, 'Require at least one z plane'
        assert self.n_yx[0] > 0, 'Require y size to be at least 1 pixel'
        assert self.n_yx[1] > 0, 'Require x size to be at least 1 pixel'
        if tiles_width == None:
            self.tiles_width = np.floor(np.sqrt(self.n_tiles))
            if self.tiles_width < 1:
                self.tiles_width = 1
        else:
            self.tiles_width = tiles_width
        assert self.tiles_width > 0, f'Require a tile width > 0, got {self.tiles_width}'
        if self.include_anchor:
            assert self.anchor_channel >= 0 and self.anchor_channel < self.n_channels, \
                f'Anchor channel must be in range 0 to {self.n_channels - 1}, but got {self.anchor_channel}'
        if self.include_dapi:
            assert self.dapi_channel >= 0 and self.dapi_channel < self.n_channels, \
                f'DAPI channel must be in range 0 to {self.n_channels - 1}, but got {self.dapi_channel}'
        if self.include_dapi:
            assert self.dapi_channel != self.anchor_channel, 'Cannot have DAPI and anchor channel identical, ' + \
                'because they are both saved in the same anchor raw file'
        # Ordering for data matrices is round x tile x channel x Y x X x Z, like the nd2 raw files for coppafish
        self.shape = (self.n_rounds, self.n_tiles, self.n_channels, self.n_yx[0], self.n_yx[1], 
            self.n_planes)
        self.anchor_shape = (self.n_tiles, self.n_channels, self.n_yx[0], self.n_yx[1], self.n_planes)
        self.presequence_shape = (self.n_tiles, self.n_channels, self.n_yx[0], self.n_yx[1], self.n_planes)
        # This is the image we will build up throughout this script and eventually return. Starting with just 
        # zeroes
        self.image = np.zeros(self.shape, dtype=self.image_dtype)
        # DAPI image is contained within the anchor image
        self.anchor_image = np.zeros(self.anchor_shape, dtype=self.image_dtype)
        self.presequence_image = np.zeros(self.presequence_shape, dtype=self.image_dtype)
        # if self.n_tiles > 1:
        #     raise NotImplementedError('Multiple tile support is not implemented yet')
        if self.n_yx[0] != self.n_yx[1]:
            raise NotImplementedError('Coppafish does not support non-square tiles')
        if self.n_yx[0] < 2_000 or self.n_yx[1] < 2_000:
            warnings.warn('Coppafish may not support tile sizes that are too small due to implicit' + \
                ' assumptions about a tile size of 2048 in the x and y directions')
        if self.n_planes < 4:
            warnings.warn('Coppafish may break with fewer than four z planes')
        
        # Calculate tile positionings
        self.tile_origins_yx = []
        self.tilepos_yx_nd2 = []
        self.xy_pos = []
        x_index = 0
        y_index = 0
        for t in range(self.n_tiles):
            self.tile_origins_yx.append([self.n_yx[0] * y_index, self.n_yx[1] * x_index])
            self.tilepos_yx_nd2.append ([y_index, x_index])
            self.xy_pos.append([self.n_yx[1] * x_index, self.n_yx[0] * y_index])
            x_index += 1
            if (t + 1) % self.tiles_width == 0:
                x_index = 0
                y_index += 1


    def Generate_Gene_Codes(self, n_genes : int = 73, n_rounds : int = None) -> Dict:
        """
        Generates random gene codes based on reed-solomon principle, using the lowest degree polynomial possible \
            based on the number of genes needed. Saves codes in self, can be used in function `Add_Spots`. The \
            `i`th gene name will be `gene_i`. `ValueError` is raised if all gene codes created are not unique. \
            We assume that n_rounds is also the number of unique dyes, each dye is labelled between \
            (0, n_rounds]. See https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction for more \
            details.

        args:
            n_genes (int, optional): Number of unique gene codes to generate. Default: 73
            n_rounds (int, optional): Number of sequencing rounds. Default: Use round number saved in `self` \
                RoboMinnie instance.

        returns:
            Dict (str:str): Gene names as keys, gene codes as values.
        """
        self.instructions.append(_funcname())
        if n_rounds == None:
            n_rounds = self.n_rounds
        assert n_rounds > 1, 'Require at least two rounds'
        assert n_genes > 0, 'Require at least one gene'
        degree = 0
        # Find the smallest degree polynomial required to produce `n_genes` unique gene codes. We use the smallest 
        # degree polynomial because this will have the smallest amount of overlap between gene codes
        while True:
            max_unique_codes = int(n_rounds**degree - n_rounds)
            if max_unique_codes >= n_genes:
                break
            degree += 1
            assert degree < 100, 'Degree too large, breaking from loop...'
        
        # Create a `degree` degree polynomial, where each coefficient goes between (0, n_rounds] to generate each 
        # unique gene code

        codes = dict()

        # Index 0 is for constant, index 1 for linear coefficient, etc..
        most_recent_coefficient_set = np.array(np.zeros(degree+1))
        for n_gene in trange(n_genes, ascii=True, unit='Codes', desc='Generating gene codes'):
            # Find the next coefficient set that works, which is not just constant across all rounds (like 
            # a background code)
            
            while True:
                # Iterate to next working coefficient set, by mod n_rounds addition
                most_recent_coefficient_set[0] += 1
                for i in range(most_recent_coefficient_set.size):
                    if most_recent_coefficient_set[i] >= n_rounds:
                        # Cycle back around to 0, then add one to next coefficient
                        most_recent_coefficient_set[i]   =  0
                        most_recent_coefficient_set[i+1] += 1
                if np.all(most_recent_coefficient_set[1:degree+1] == 0):
                    continue
                break

            # Generate new gene code
            new_code  = ''
            gene_name = f'gene_{n_gene}'
            for r in range(n_rounds):
                result = 0
                for j in range(degree + 1):
                    result += most_recent_coefficient_set[j] * r**j
                result = int(result)
                result %= n_rounds
                new_code += str(result)
            # Add new code to dictionary
            codes[gene_name] = new_code
        values = list(codes.values())
        if len(values) != len(set(values)):
            # Not every gene code is unique
            raise ValueError(f'Could not generate {n_genes} unique gene codes with {n_rounds} rounds/dyes. ' + \
                             'Maybe try decreasing the number of genes or increasing the number of rounds.')
        self.codes = codes
        return codes


    def Generate_Pink_Noise(self, noise_amplitude : float, noise_spatial_scale : float) -> None:
        """
        Superimpose pink noise onto all images, including the anchor, DAPI and pre-sequence images, if used. The \
        noise is identical on all images because pink noise is a good estimation for biological things that \
        fluoresce. See https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.814 for more details.

        args:
            noise_amplitude (float): The maximum possible noise intensity
            noise_spatial_scale (float): Spatial scale of noise
        """
        #TODO: Add the ability to square the pink noise, gives sharper peaks which is meant to be more realistic
        self.instructions.append(_funcname())
        print('Generating pink noise')

        from numpy.fft import fftshift, ifftshift
        from scipy.fft import fftn, ifftn

        # True spatial scale should be maintained regardless of the image size, so we
        # scale it as such.
        true_noise_spatial_scale = noise_spatial_scale *  np.asarray([*self.n_yx,10*self.n_planes])
        # Generate pink noise
        pink_spectrum = 1 / (
            1 + np.linspace(0, true_noise_spatial_scale[0],  self.n_yx[0]) [:,None,None]**2 +
            np.linspace(0,     true_noise_spatial_scale[1],  self.n_yx[1]) [None,:,None]**2 +
            np.linspace(0,     true_noise_spatial_scale[2],  self.n_planes)[None,None,:]**2
        )
        rng = np.random.RandomState(self.seed)
        pink_sampled_spectrum = pink_spectrum*fftshift(fftn(rng.randn(*self.n_yx, self.n_planes)))
        pink_noise = np.abs(ifftn(ifftshift(pink_sampled_spectrum)))
        pink_noise = (pink_noise - np.mean(pink_noise))/np.std(pink_noise) * noise_amplitude
        self.image[:,:,:] += pink_noise
        if self.include_anchor:
            self.anchor_image[:,self.anchor_channel] += pink_noise
        if self.include_presequence:
            self.presequence_image[:,:] += pink_noise
        if self.include_dapi:
            pink_sampled_spectrum = pink_spectrum*fftshift(fftn(rng.randn(*self.n_yx, self.n_planes)))
            pink_noise = np.abs(ifftn(ifftshift(pink_sampled_spectrum)))
            pink_noise = (pink_noise - np.mean(pink_noise))/np.std(pink_noise) * noise_amplitude
            self.anchor_image[:,self.dapi_channel] += pink_noise


    def Generate_Random_Noise(self, noise_std : float, noise_mean_amplitude : float = 0, 
        noise_type : str = 'normal', include_anchor : bool = True, include_presequence : bool = True, 
        include_dapi : bool = True) -> None:
        """
        Superimpose random, white noise onto every pixel individually. Good for modelling random noise from the 
        camera

        args:
            noise_mean_amplitude (float, optional): Mean amplitude of random noise. Default: 0
            noise_std (float): Standard deviation/width of random noise
            noise_type (str('normal' or 'uniform'), optional): Type of random noise to apply. Default: 'normal'
            include_anchor (bool, optional): Whether to apply random noise to anchor image. Default: true
            include_presequence (bool, optional): Whether to apply random noise to presequence rounds. Default: true
            include_dapi (bool, optional): Whether to apply random noise to DAPI image. Default: true
        """
        self.instructions.append(_funcname())

        assert noise_std > 0, f'Noise standard deviation must be > 0, got {noise_std}'

        rng = np.random.RandomState(self.seed)
        anchor_shape_single_tile = (1, *self.anchor_shape[1:])
        shape_single_tile = (self.shape[0], 1, *self.shape[2:])
        presequence_shape_single_tile = (1, *self.presequence_shape[1:])


        def _generate_noise(_rng : Generator, _noise_type : str, _noise_mean_amplitude : float, 
                            _noise_std : float, _size : tuple) -> npt.NDArray:
            """
            Generate noise based on the specified noise type and parameters.

            Args:
                _rng (Generator): Random number generator.
                _noise_type (str): Type of noise ('normal' or 'uniform').
                _noise_mean_amplitude (float): Mean or center of the noise distribution.
                _noise_std (float): Standard deviation or spread of the noise distribution.
                size (tuple of int, int, int): Size of the noise array.

            Returns:
                ndarray: Generated noise array.

            Raises:
                ValueError: If an unsupported noise type is provided.
            """
            if _noise_type == 'normal':
                return _rng.normal(_noise_mean_amplitude, _noise_std, _size)
            elif _noise_type == 'uniform':
                return _rng.uniform(_noise_mean_amplitude - _noise_std/2, _noise_mean_amplitude + _noise_std/2, _size)
            else:
                raise ValueError(f'Unknown noise type: {_noise_type}')


        for t in trange(self.n_tiles, ascii=True, desc=f'Generating random noise', unit='tiles'):
            anchor_noise = np.zeros(shape=anchor_shape_single_tile)
            presequence_noise = np.zeros(shape=presequence_shape_single_tile)

            noise = _generate_noise(rng, noise_type, noise_mean_amplitude, noise_std, shape_single_tile)

            if include_anchor and self.include_anchor:
                anchor_noise[0, self.anchor_channel] = \
                    _generate_noise(
                        rng, noise_type, noise_mean_amplitude, noise_std, anchor_shape_single_tile
                    )[0, self.anchor_channel]
            if include_presequence and self.include_presequence:
                presequence_noise = \
                    _generate_noise(
                        rng, noise_type, noise_mean_amplitude, noise_std, presequence_shape_single_tile
                    )
            if include_dapi and self.include_dapi:
                anchor_noise[0, self.dapi_channel] = \
                    _generate_noise(
                        rng, noise_type, noise_mean_amplitude, noise_std, anchor_shape_single_tile
                    )[0, self.dapi_channel]
            
            # Add new random noise to each pixel
            np.add(self.image[:,t], noise[:,0], out=self.image[:,t])
            np.add(self.anchor_image[t], anchor_noise[0], out=self.anchor_image[t])
            np.add(self.presequence_image[t], presequence_noise[0], out=self.presequence_image[t])
            del noise
            del anchor_noise
            del presequence_noise


    def Add_Spots(self, n_spots : int, bleed_matrix : npt.NDArray, spot_size_pixels : npt.NDArray, \
        gene_codebook_path : str = USE_INSTANCE_GENE_CODES) -> None:
        """
        Superimpose spots onto images in both space and channels (based on the bleed matrix). Also applied to the 
        anchor when included. The spots are uniformly, randomly distributed across each image. Never applied to 
        presequence images.

        args:
            n_spots (int): Number of spots to add
            bleed_matrix (n_dyes x n_channels ndarray[float, float]): The bleed matrix, used to map each dye to 
            its pattern as viewed by the camera in each channel.
            spot_size_pixels (ndarray[float, float, float]): The spot's standard deviation in directions x, y, z \
            respectively.
            gene_codebook_path (str): Path to the gene codebook, saved as a .txt file. Default: use `self` gene \
                codes instead, which can be generated by calling `Generate_Gene_Codes`
        """
        self.instructions.append(_funcname())
        
        def blit(source, target, loc):
            """
            Superimpose given spot image (source) onto a target image (target) at the centred position loc. The 
            parameter target is then updated with the final image.

            args:
                source (n_channels (optional) x spot_size_y x spot_size_x x spot_size_z ndarray): The spot image
                target (n_channels (optional) x tile_size_y x tile_size_x x tile_size_z ndarray): The tile image
                loc (channel (optional), y, x, z ndarray): Centre location of spot
            """
            source_size = np.asarray(source.shape)
            target_size = np.asarray(target.shape)
            # If we had infinite boundaries, where would we put it?  Assume "loc" is the centre of "target"
            target_loc_tl = loc - source_size//2
            target_loc_br = target_loc_tl + source_size
            # Compute the index for the source
            source_loc_tl = -np.minimum(0, target_loc_tl)
            source_loc_br = source_size - np.maximum(0, target_loc_br - target_size)
            # Recompute the index for the target
            target_loc_br = np.minimum(target_size, target_loc_tl+source_size)
            target_loc_tl = np.maximum(0, target_loc_tl)
            # Compute slices from positions
            target_slices = [slice(s1, s2) for s1,s2 in zip(target_loc_tl,target_loc_br)]
            source_slices = [slice(s1, s2) for s1,s2 in zip(source_loc_tl,source_loc_br)]
            # Perform the blit
            target[tuple(target_slices)] += source[tuple(source_slices)]
        
        assert bleed_matrix.shape[1] == self.n_channels, \
            f'Bleed matrix does not have n_channels={self.n_channels} as expected'
        assert n_spots > 0, f'Expected n_spots > 0, got {n_spots}'
        if gene_codebook_path != USE_INSTANCE_GENE_CODES:
            assert os.path.isfile(gene_codebook_path), f'Gene codebook at {gene_codebook_path} does not exist'
        assert spot_size_pixels.size == 3, 'Spot size must be in three dimensions'
        if bleed_matrix.shape[0] != bleed_matrix.shape[1]:
            warnings.warn(f'Given bleed matrix does not have equal channel and dye counts like usual')
        if self.bleed_matrix is None:
            self.bleed_matrix = bleed_matrix
        else:
            assert self.bleed_matrix == bleed_matrix, 'All added spots must have the same shared bleed matrix'

        self.n_spots += n_spots

        if gene_codebook_path != USE_INSTANCE_GENE_CODES:
            # Read in the gene codebook txt file
            _codes = dict()
            with open(gene_codebook_path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc='Reading genebook', ascii=True, unit='genes'):
                    if not line:
                        # Skip empty lines
                        continue
                    phrases = line.split()
                    gene, code = phrases[0], phrases[1]
                    # Save the gene name as a key, the value is the gene's code
                    _codes[gene] = code
            self.codes = _codes

        values = list(self.codes.values())
        assert len(values) == len(set(values)), f'Duplicate gene code found in dictionary: {self.codes}'

        # Generate the random spots
        rng = np.random.RandomState(self.seed)
        true_spot_positions_pixels = rng.rand(n_spots, 3) * [*self.n_yx, self.n_planes]
        true_spot_identities = list(rng.choice(list(self.codes.keys()), n_spots))

        # We assume each spot is a multivariate gaussian with a diagonal covariance,
        # where variance in each dimension is given by the spot size.  We create a spot
        # template image and then iterate through spots.  Each iteration, we add
        # ("blit") the spot onto the image such that the centre of the spot is in the
        # middle.  The size of the spot template is guaranteed to be odd, and is about
        # 1.5 times the standard deviation.  We add it to the appropriate color channels
        # (by transforming through the bleed matrix) and then also add the spot to the
        # anchor.
        ind_size = np.ceil(spot_size_pixels*1.5).astype(int)*2+1
        indices = np.indices(ind_size)-ind_size[:,None,None,None]//2
        spot_img = \
            scipy.stats.multivariate_normal([0, 0, 0], np.eye(3)*spot_size_pixels).pdf(indices.transpose(1,2,3,0))
        # TODO: Add spots to multiple tiles, not just zeroth tile
        for p,ident in tqdm(zip(true_spot_positions_pixels, true_spot_identities), \
                            desc='Superimposing spots', ascii=True, unit='spots', total=n_spots):
            p = np.asarray(p).astype(int)
            p_chan = np.round([self.n_channels//2, p[0], p[1], p[2]]).astype(int)
            for r in range(self.n_rounds):
                dye = int(self.codes[ident][r])
                # The second index on image is for one tile
                blit(spot_img[None,:] * bleed_matrix[dye][:,None,None,None], self.image[r,0], p_chan)
            blit(spot_img, self.anchor_image[0,self.anchor_channel], p)

        # Append just in case spots are created multiple times
        self.true_spot_identities = np.append(self.true_spot_identities, np.array(true_spot_identities))
        self.true_spot_positions_pixels = np.append(self.true_spot_positions_pixels, true_spot_positions_pixels, axis=0)


    # Post-Processing function
    def Fix_Image_Minimum(self, minimum : float = 0.) -> None:
        """
        Ensure all pixels in the images are greater than or equal to given value (minimum). Includes the \
            presequence and anchor images, if they exist.

        args:
            minimum (float, optional): Minimum pixel value allowed. Default: 0
        """
        self.instructions.append(_funcname())
        print(f'Fixing image minima')

        minval = self.image.min()
        minval = \
            np.min([minval, self.anchor_image[0,self.anchor_channel].min() if self.include_anchor else np.inf])
        minval = \
            np.min([minval, self.presequence_image.min() if self.include_presequence              else np.inf])
        minval = \
            np.min([minval, self.anchor_image[0,self.dapi_channel].min() if self.include_dapi     else np.inf])
        offset = -(minval + minimum)
        self.image += offset
        if self.include_anchor:
            self.anchor_image += offset
        if self.include_presequence:
            self.presequence_image += offset
    

    # Post-Processing function
    def Offset_Images_By(self, constant : float, include_anchor : bool = True, include_presequence : bool = True, 
                         include_dapi : bool = True) -> None:
        """
        Shift every image pixel, in all tiles, by a constant value

        args:
            constant (float): Shift value
            include_anchor (bool, optional): Include anchor image in the shift, if exists. Default: true.
            include_presequence (bool, optional): Include preseq images in the shift, if exists. Default: true.
            include_dapi (bool, optional): Offset DAPI image, if exists. Default: true.
        """
        self.instructions.append(_funcname())
        print(f'Shifting image by {constant}')

        np.add(self.image, constant, out=self.image)
        if include_anchor and self.include_anchor:
            np.add(
                self.anchor_image[0,self.anchor_channel], constant, out=self.anchor_image[:,self.anchor_channel]
            )
        if include_presequence and self.include_presequence:
            np.add(self.presequence_image, constant, out=self.presequence_image)
        if include_dapi and self.include_dapi:
            np.add(
                self.anchor_image[0,self.dapi_channel], constant, out=self.anchor_image[:,self.dapi_channel]
            )


    def Save_Raw_Images(self, output_dir : str, overwrite : bool = False, omp_iterations : int = 5, \
                        omp_initial_intensity_thresh_percentile : int = 25) -> None:
        """
        Save known spot positions and codes, raw .npy image files, metadata.json file, gene codebook and \
            config.ini file for coppafish pipeline run. Output directory must be empty. After saving, able to \
            call function `Run_Coppafish` to run the coppafish pipeline.
        
        args:
            output_dir (str): Save directory
            overwrite (bool, optional): If True, overwrite any saved coppafish data inside the directory, \
                delete old notebook.npz file if there is one and ignore any other files inside the directory
            omp_iterations (int, optional): Number of OMP iterations on every pixel. Increasing this may improve \
                gene scoring. Default: 5
            omp_initial_intensity_thresh_percentile (float, optional): percentile of the absolute intensity of \
                all pixels in the mid z-plane of the central tile. Used as a threshold for pixels to decide what \
                to apply OMP on. A higher number leads to stricter picking of pixels. Default: 25, the default \
                coppafish value
        """
        self.instructions.append(_funcname())
        print(f'Saving as coppafish data')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        if not overwrite:
            assert len(os.listdir(output_dir)) == 0, f'Output directory {output_dir} must be empty'

        # Create an output_dir/output_coppafish directory for coppafish pipeline output saved to disk
        self.coppafish_output = os.path.join(output_dir, 'output_coppafish')
        if not os.path.isdir(self.coppafish_output):
            os.mkdir(self.coppafish_output)
        if overwrite:
            # Delete all files located in coppafish output directory to stop coppafish using old data
            for filename in os.listdir(self.coppafish_output):
                filepath = os.path.join(self.coppafish_output, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)

        # Create an output_dir/output_coppafish/tiles directory for coppafish extract output
        self.coppafish_tiles = os.path.join(self.coppafish_output, 'tiles')
        if not os.path.isdir(self.coppafish_tiles):
            os.mkdir(self.coppafish_tiles)
        # Remove any old tile files in the tile directory, if any, to make sure coppafish runs extract and filter \
        # again
        for filename in os.listdir(self.coppafish_tiles):
            filepath = os.path.join(self.coppafish_tiles, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)

        # Save the known gene names and positions to a csv.
        df = pandas.DataFrame(
            {
                "gene": self.true_spot_identities, 
                "z": self.true_spot_positions_pixels[:,0],
                "y": self.true_spot_positions_pixels[:,1],
                "x": self.true_spot_positions_pixels[:,2]
            }
        )
        df.to_csv(os.path.join(output_dir, "gene_locations.csv"))

        #TODO: Update metadata entries for multiple tile support, fix this
        metadata = {
            "n_tiles": self.n_tiles, 
            "n_rounds": self.n_rounds, 
            "n_channels": self.n_channels, 
            "tile_sz": self.n_yx[0], 
            "pixel_size_xy": 0.26, 
            "pixel_size_z": 0.9, 
            "tile_centre": [self.n_yx[0]//2,self.n_yx[1]//2,self.n_planes//2], 
            "tilepos_yx": self.tile_origins_yx, 
            "tilepos_yx_nd2": self.tilepos_yx_nd2, 
            "channel_camera": [1,1,2,3,2,3,3], 
            "channel_laser": [1,1,2,3,2,3,3], 
            "xy_pos": self.xy_pos, 
        }
        self.metadata_filepath = os.path.join(output_dir, 'metadata.json')
        with open(self.metadata_filepath, 'w') as f:
            json.dump(metadata, f)

        # Save the raw .npy files, one round at a time, in separate round directories. We do this because 
        # coppafish expects every rounds (and anchor and presequence) in its own directory.
        all_images = self.image
        presequence_directory = f'presequence'
        dask_chunks = self.shape[1:]
        # Dask should save each tile as a separate .npy file for coppafish to read
        dask_chunks = list(dask_chunks)
        dask_chunks[0] = 1
        dask_chunks = tuple(dask_chunks)
        if self.include_anchor or self.include_dapi:
            # Add a new dimension for the round axis
            all_images = np.concatenate([self.image, self.anchor_image[None]]).astype(self.image_dtype)
        for r in range(self.n_rounds + self.include_anchor):
            save_path = os.path.join(output_dir, f'{r}')
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            # Clear the raw .npy directories before dask saving, so old multi-tile data is not left in the 
            # directories
            for filename in os.listdir(save_path):
                filepath = os.path.join(save_path, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                else:
                    raise IsADirectoryError(f'Found unexpected directory in {save_path}')
            image_dask = dask.array.from_array(all_images[r], chunks=dask_chunks)
            dask.array.to_npy_stack(save_path, image_dask)
            del image_dask
        if self.include_presequence:
            # Save the presequence image in `coppafish_output/presequence/`
            presequence_save_path = os.path.join(output_dir, presequence_directory)
            if not os.path.isdir(presequence_save_path):
                os.mkdir(presequence_save_path)
            for filename in os.listdir(presequence_save_path):
                filepath = os.path.join(presequence_save_path, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                else:
                    raise IsADirectoryError(f'Found unexpected directory in {presequence_save_path}')
            image_dask = dask.array.from_array(self.presequence_image.astype(self.image_dtype), chunks=dask_chunks)
            dask.array.to_npy_stack(presequence_save_path, image_dask)
            del image_dask
        del all_images

        # Save the gene codebook in `coppafish_output`
        self.codebook_filepath = os.path.join(output_dir, 'codebook.txt')
        with open(self.codebook_filepath, 'w') as f:
            for genename, code in self.codes.items():
                f.write(f'{genename} {code}\n')

        # Save the initial bleed matrix for the config file
        self.initial_bleed_matrix_filepath = os.path.join(output_dir, 'bleed_matrix.npy')
        np.save(self.initial_bleed_matrix_filepath, self.bleed_matrix)

        # Save the config file. z_subvols is moved from the default of 5 based on n_planes.
        #! Note: Older coppafish software used to take settings ['register']['z_box'] for each dimension 
        # subvolume. Newer coppafish software (supported here) uses ['register']['box_size'] to change subvolume 
        # size in each dimension in one setting
        self.dye_names = map(''.join, zip(['dye_']*self.n_channels, list(np.arange(self.n_channels).astype(str))))
        self.dye_names = list(self.dye_names)

        config_file_contents = f"""; This config file is auto-generated by RoboMinnie. 
        [file_names]
        notebook_name = notebook.npz
        input_dir = {output_dir}
        output_dir = {self.coppafish_output}
        tile_dir = {self.coppafish_tiles}
        initial_bleed_matrix = {self.initial_bleed_matrix_filepath}
        round = {', '.join([str(i) for i in range(self.n_rounds)])}
        anchor = {self.n_rounds if self.include_anchor else ''}
        pre_seq = {presequence_directory if self.include_presequence else ''}
        code_book = {self.codebook_filepath}
        raw_extension = .npy
        raw_metadata = {self.metadata_filepath}

        [basic_info]
        is_3d = True
        dye_names = {', '.join(self.dye_names)}
        use_rounds = {', '.join([str(i) for i in range(self.n_rounds)])}
        use_z = {', '.join([str(i) for i in range(self.n_planes)])}
        use_tiles = {', '.join(str(i) for i in range(self.n_tiles))}
        anchor_round = {self.n_rounds if self.include_anchor else ''}
        anchor_channel = {self.anchor_channel if self.include_anchor else ''}
        dapi_channel = {self.dapi_channel if self.include_dapi else ''}
        
        [extract]
        r_dapi = {48 if self.include_dapi else ''}

        [register]
        subvols = {1}, {8}, {8}
        box_size = {np.min([self.n_planes, 12])}, 300, 300
        pearson_r_thresh = 0.25

        [omp]
        max_genes = {omp_iterations}
        initial_intensity_thresh_percentile = {omp_initial_intensity_thresh_percentile}
        """
        # Remove any large spaces in the config contents
        config_file_contents = config_file_contents.replace('  ', '')

        self.config_filepath = os.path.join(output_dir, 'config.ini')
        with open(self.config_filepath, 'w') as f:
            f.write(config_file_contents)

        # Save the instructions run so far (every function call)
        self.instruction_filepath = os.path.join(output_dir, 'instructions.txt')
        with open(self.instruction_filepath, 'w') as f:
            for instruction in self.instructions:
                f.write(instruction + '\n')


    def Run_Coppafish(self, time_pipeline : bool = True, include_omp : bool = True, \
        jax_profile_omp : bool = False, save_ref_spots_data : bool = False) -> None:
        """
        Run RoboMinnie instance on the entire coppafish pipeline.

        args:
            time_pipeline (bool, optional): If true, print the time taken to run the coppafish pipeline. Default: \
                true
            include_omp (bool, optional): If true, run up to and including coppafish OMP stage
            jax_profile_omp (bool, optional): If true, profile coppafish OMP using the jax tensorboard profiler. \
                Requires tensorflow, install by running `pip install tensorflow tensorboard-plugin-profile`. \
                Default: false
            save_ref_spots_data (bool, optional): If true, will save ref_spots data, which is used for comparing \
                ref_spots results to the true robominnie spots. Default: false to reduce RoboMinnie's memory usage
        """
        self.instructions.append(_funcname())
        print(f'Running coppafish')
        from coppafish.pipeline.run import initialize_nb, run_tile_indep_pipeline, run_stitch, \
            run_reference_spots, run_omp
        if jax_profile_omp:
            import jax
        
        # Run the non-parallel pipeline code\
        nb = initialize_nb(self.config_filepath)
        if time_pipeline:
            start_time = time.time()
        run_tile_indep_pipeline(nb)
        run_stitch(nb)

        assert nb.stitch is not None, f'Stitch not found in notebook at {self.config_filepath}'
        # Keep the stitch information to convert local tiles coordinates into global coordinates when comparing 
        # to true spots
        self.stitch_tile_origins = nb.stitch.tile_origin
        run_reference_spots(nb, overwrite_ref_spots=False)

        # Keep reference spot information to compare to true spots, if wanted
        assert nb.ref_spots is not None, f'Reference spots not found in notebook at {self.config_filepath}'
        if save_ref_spots_data:
            self.ref_spots_scores              = nb.ref_spots.score
            self.ref_spots_local_positions_yxz = nb.ref_spots.local_yxz
            self.ref_spots_intensities         = nb.ref_spots.intensity
            self.ref_spots_gene_indices        = nb.ref_spots.gene_no
            self.ref_spots_tile                = nb.ref_spots.tile

        if include_omp == False:
            return
        
        if jax_profile_omp:
            jax.profiler.start_trace(self.coppafish_output, create_perfetto_link=True, create_perfetto_trace=True)
        run_omp(nb)
        if jax_profile_omp:
            jax.profiler.stop_trace()
        if time_pipeline:
            end_time = time.time()
            print(
                f'Coppafish pipeline run: {round(end_time - start_time, 1)}s. ' + \
                f'{round((end_time - start_time)//(self.n_planes * self.n_tiles), 1)}s per z plane per tile.'
            )

        assert nb.omp is not None, f'OMP not found in notebook at {self.config_filepath}'
        # Keep the OMP spot intensities, assigned gene, assigned tile number and the spot positions in the class instance
        self.omp_spot_intensities = nb.omp.intensity
        self.omp_gene_numbers = nb.omp.gene_no
        self.omp_tile_number = nb.omp.tile
        self.omp_spot_local_positions = nb.omp.local_yxz # yxz position of each gene found
        assert self.omp_gene_numbers.shape[0] == self.omp_spot_local_positions.shape[0], \
            'Mismatch in spot count in omp.gene_numbers and omp.local_positions'
        self.omp_spot_count = self.omp_gene_numbers.shape[0]

        if self.omp_spot_count == 0:
            warnings.warn('Copppafish OMP found zero spots')


    def Compare_Ref_Spots(self, score_threshold : float = 0.5, intensity_threshold : float = 0.7, \
        location_threshold : float = 2) -> Tuple[int,int,int,int]:
        """
        Compare spot positions and gene codes from coppafish ref_spots results to the known spot locations. If \
            the spots are close enough and the true spot has not been already assigned to a reference spot, then \
            they are considered the same spot in both coppafish output and synthetic data. If two or more spots \
            are close enough to a true spot, then the closest one is chosen. If equidistant, then take the spot \
            with the correct gene code. If not applicable, then just take the spot with the lowest index \
            (effectively choose one of them at random, but consistent way). Will save the results in `self` \
            (overwrites other earlier comparison calls), so can call the `Overall_Score` function afterwards \
            without inputting parameters.

        args:
            score_threshold (float, optional): Reference spot score threshold, any spots below this intensity are \
                ignored. Default: 0.5
            intensity_threshold (float, optional): Reference spot intensity threshold. Default: 0.7
            location_threshold (float, optional): Max distance two spots can be apart to be considered the same \
                spot in pixels, inclusive. Default: 2

        returns:
            Tuple (true_positives : int, wrong_positives : int, false_positives : int, false_negatives : int): \
                The number of spots assigned to true positive, wrong positive, false positive and false negative \
                respectively, where a wrong positive is a spot assigned to the wrong gene, but found in the \
                location of a true spot
        """
        assert 'Run_Coppafish' in self.instructions, \
            'Run_Coppafish must be called before comparing reference spots to ground truth spots'

        self.instructions.append(_funcname())
        print(f'Comparing reference spots to known spots')

        assert score_threshold >= 0 and score_threshold <= 1, \
            f'Intensity threshold must be (0,1), got {score_threshold}'
        assert location_threshold >= 0, f'Location threshold must be >= 0, got {location_threshold}'

        location_threshold_squared = location_threshold ** 2

        # Convert local spot positions into global coordinates using global tile positions
        ref_spots_positions_yxz = self.ref_spots_local_positions_yxz.astype(np.float64)
        np.add(ref_spots_positions_yxz, self.stitch_tile_origins[self.ref_spots_tile], out=ref_spots_positions_yxz)

        ref_spots_gene_indices = self.ref_spots_gene_indices
        ref_spots_intensities  = self.ref_spots_intensities
        ref_spots_scores       = self.ref_spots_scores

        # Eliminate any reference spots below the thresholds
        indices = np.logical_and(ref_spots_intensities >= intensity_threshold, ref_spots_scores > score_threshold)
        ref_spots_gene_indices = ref_spots_gene_indices[indices]
        ref_spots_positions_yxz = ref_spots_positions_yxz[indices]
        del indices

        # Note: self.true_spot_positions_pixels and ref_spots_positions_yxz has the form yxz
        true_positives, wrong_positives, false_positives, false_negatives = \
            _compare_spots(
                ref_spots_positions_yxz,
                ref_spots_gene_indices,
                self.true_spot_positions_pixels,
                self.true_spot_identities,
                location_threshold_squared,
                self.codes,
                'Checking reference spots',
            )
        # Save results in `self` (overwrites)
        self.true_positives  = true_positives
        self.wrong_positives = wrong_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        return (true_positives, wrong_positives, false_positives, false_negatives)


    def Compare_OMP_Spots(self, omp_intensity_threshold : float = 0.2, location_threshold : float = 2) \
        -> Tuple[int,int,int,int]:
        """
        Compare spot positions and gene codes from coppafish OMP results to the known spot locations. If the spots 
        are close enough and the true spot has not been already assigned to an omp spot, then they are considered 
        the same spot in both coppafish output and synthetic data. If two or more spots are close enough to a true 
        spot, then the closest one is chosen. If equidistant, then take the spot with the correct gene code. If 
        not applicable, then just take the spot with the lowest index (effectively choose one of them at random, 
        but consistent way). Will save the results in `self` (overwrites other earlier comparison calls), so can 
        call the `Overall_Score` function afterwards without inputting parameters.

        args:
            omp_intensity_threshold (float, optional): OMP intensity threshold, any spots below this intensity \
                are ignored. Default: 0.2
            location_threshold (float, optional): Max distance two spots can be apart to be considered the same \
                spot in pixels, inclusive. Default: 2

        returns:
            Tuple (true_positives : int, wrong_positives : int, false_positives : int, false_negatives : int): \
                The number of spots assigned to true positive, wrong positive, false positive and false negative \
                respectively, where a wrong positive is a spot assigned to the wrong gene, but found in the \
                location of a true spot
        """
        assert 'Run_Coppafish' in self.instructions, \
            'Run_Coppafish must be called before comparing OMP spots to ground truth spots'

        self.instructions.append(_funcname())
        print(f'Comparing OMP spots to known spots')

        assert omp_intensity_threshold >= 0 and omp_intensity_threshold <= 1, \
            f'Intensity threshold must be between 0 and 1, got {omp_intensity_threshold}'
        assert location_threshold >= 0, f'Location threshold must be >= 0, got {location_threshold}'

        location_threshold_squared = location_threshold ** 2

        # Convert local spot positions into global coordinates using global tile positions
        omp_spot_positions_yxz = self.omp_spot_local_positions.astype(np.float64)
        np.add(omp_spot_positions_yxz, self.stitch_tile_origins[self.omp_tile_number], out=omp_spot_positions_yxz)

        omp_gene_numbers = self.omp_gene_numbers

        # Eliminate OMP spots below threshold
        indices = self.omp_spot_intensities >= omp_intensity_threshold
        omp_gene_numbers = omp_gene_numbers[indices]
        omp_spot_positions_yxz = omp_spot_positions_yxz[indices]
        del indices

        # Note: self.true_spot_positions_pixels and omp_spot_positions has the form yxz
        true_positives, wrong_positives, false_positives, false_negatives = \
            _compare_spots(
                omp_spot_positions_yxz,
                omp_gene_numbers,
                self.true_spot_positions_pixels,
                self.true_spot_identities,
                location_threshold_squared,
                self.codes,
                'Checking OMP spots',
            )
        # Save results in `self` (overwrites)
        self.true_positives  = true_positives
        self.wrong_positives = wrong_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        return (true_positives, wrong_positives, false_positives, false_negatives)


    def Overall_Score(self, true_positives : int = None, wrong_positives : int = None, false_positives : int = None, \
        false_negatives : int = None) -> float:
        """
        Overall score from a spot-to-spot comparison, such as `Compare_OMP_Spots`.

        args:
            true_positives (int, optional): True positives spot count. Default: value stored in `self`
            wrong_positives (int, optional): Wrong positives spot count. Default: value stored in `self`
            false_positives (int, optional): False positives spot count. Default: value stored in `self`
            false_negatives (int, optional): False negatives spot count. Default: value stored in `self`

        returns:
            float: Overall score
        """
        if true_positives == None:
            true_positives  = self.true_positives
        if wrong_positives == None:
            wrong_positives = self.wrong_positives
        if false_positives == None:
            false_positives = self.false_positives
        if false_negatives == None:
            false_negatives = self.false_negatives
        return float(true_positives / (true_positives + wrong_positives + false_positives + false_negatives))


    # Debugging Function:
    def View_Images(self, tiles : List = [0]):
        """
        View all images in `napari` for tile index `t`, including a presequence and anchor image, if they exist.

        args:
            tile (list of int, optional): Tile index. Default: [0]
        """
        viewer = napari.Viewer(title=f'RoboMinnie, tiles={tiles}')
        for t in tiles:
            for c in range(self.n_channels):
                for r in range(self.n_rounds):
                    # z index must be the first axis for napari to view
                    viewer.add_image(self.image[r,t,c].transpose([2,0,1]), name=f't={t}, r={r}, c={c}', visible=False)
                if self.include_presequence:
                    viewer.add_image(
                        self.presequence_image[t,c].transpose([2,0,1]), 
                        name=f'preseq, t={t}, c={c}',
                        visible=False,
                    )
            if self.include_anchor:
                viewer.add_image(
                    self.anchor_image[t,self.anchor_channel].transpose([2,0,1]), 
                    name=f'anchor, t={t}, c={c}', 
                    visible=False
                )
            if self.include_dapi:
                viewer.add_image(
                    self.anchor_image[t,self.dapi_channel].transpose([2,0,1]), 
                    name=f'dapi, t={t}, c={c}', 
                    visible=False
                )
        # viewer.show()
        napari.run()


    def Save(self, output_dir : str, filename : str = None, compress : bool = False) -> None:
        """
        Save `RoboMinnie` instance using the amazing tool pickle inside output_dir directory.

        args:
            output_dir (str): 
            filename (str, optional): Name of the pickled `RoboMinnie` object. Default: 'robominnie.pkl'
            compress (bool, optional): If True, compress pickle binary file using bzip2 compression in the \
                default python package `bz2`. Default: False
        """
        self.instructions.append(_funcname())
        print('Saving RoboMinnie instance')
        instance_output_dir = output_dir
        if filename == None:
            instance_filename = DEFAULT_INSTANCE_FILENAME
        else:
            instance_filename = filename
        if not os.path.isdir(instance_output_dir):
            os.mkdir(instance_output_dir)

        instance_filepath = os.path.join(instance_output_dir, instance_filename)
        assert not os.path.isfile(instance_filepath), \
            f'RoboMinnie instance already saved as {instance_filepath}'

        if not compress:
            with open(instance_filepath, 'wb') as f:
                pickle.dump(self, f)
        else:
            with bz2.open(instance_filepath, 'wb', compresslevel=9) as f:
                pickle.dump(self, f)


    def Load(self, input_dir : str, filename : str = None, overwrite_self : bool = True, compressed : bool = False):
        """
        Load `RoboMinnie` instance using the handy pickled information saved inside input_dir.

        args:
            input_dir (str): The directory where the RoboMinnie data is stored
            filename (str, optional): Name of the pickle RoboMinnie object. Default: 'robominnie.pkl'
            overwrite_self (bool, optional): If true, become the RoboMinnie instance loaded from disk
            compressed (bool, optional): If True, try decompress pickle binary file assuming a bzip2 compression. \
                Default: False

        Returns:
            Loaded `RoboMinnie` class
        """
        self.instructions.append(_funcname())
        print('Loading RoboMinnie instance')
        instance_input_dir = input_dir
        if filename == None:
            instance_filename = DEFAULT_INSTANCE_FILENAME
        else:
            instance_filename = filename
        instance_filepath = os.path.join(instance_input_dir, instance_filename)
        
        assert os.path.isfile(instance_filepath), f'RoboMinnie instance not found at {instance_filepath}'

        if not compressed:
            with open(instance_filepath, 'rb') as f:
                instance : RoboMinnie = pickle.load(f)
        else:
            with bz2.open(instance_filepath, 'rb') as f:
                instance : RoboMinnie = pickle.load(f)

        if overwrite_self:
            self = instance
        return instance
