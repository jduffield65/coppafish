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


DEFAULT_INSTANCE_FILENAME = 'robominnie.pkl'


def _funcname():
    return str(inspect.stack()[1][3])


class RoboMinnie:
    """
    RoboMinnie
    ==========
    The fastest coppafish integration testing suite
    
    Provides:
    ---------
    1. Single tile, modular, customisable synthetic data generation for coppafish
    2. Coppafish raw .npy file generation for pipeline running
    3. Coppafish scoring using ground-truth spot data

    Usage:
    ------
    Create new RoboMinnie instance for each integration test. Call functions for data generation (see \
        ``RoboMinnie.py`` functions for choices). Call ``Save_Coppafish`` then ``Run_Coppafish``. Use \
        ``Compare_Spots_OMP`` to evaluate OMP results.
    """
    #TODO: Multi-tile support
    #TODO: Presequence support
    def __init__(self, n_channels : int = 7, n_tiles : int = 1, n_rounds : int = 7, n_planes : int = 4, 
        n_yx : Tuple[int, int] = (2048, 2048), include_anchor : bool = True, include_preseq : bool = False, 
        anchor_channel : int = 0, seed : int = 0) -> None:
        """
        Create a new RoboMinnie instance. Used to manipulate, create and save synthetic data in a modular,
        customisable way. Synthetic data is saved as raw .npy files and includes everything needed for coppafish 
        to run a full pipeline.

        args:
            n_channels (int, optional): Number of channels. Default: 7
            n_tiles (int, optional): Number of tiles. Default: 1
            n_rounds (int, optional): Number of rounds. Not including anchor or pre-sequencing. Default: 7
            n_planes (int, optional): Number of z planes. Default: 4
            n_yx (Tuple[int, int], optional): Number of pixels for each tile in the y and x directions \
                respectively. Default: (2048, 2048).
            include_anchor (bool, optional): Whether to include the anchor round. Default:  true
            include_preseq (bool, optional): Whether to include the pre-sequence round. Default: false
            anchor_channel (int, optional): The anchor channel. Default: 0.
            seed (int, optional): Seed used throughout the generation of random data. Default: 0 for \
                reproducible output.
        """
        self.n_channels = n_channels
        self.n_tiles = n_tiles
        self.n_rounds = n_rounds
        self.n_planes = n_planes
        self.n_yx = n_yx
        self.n_spots = 0
        self.true_spot_positions_pixels = np.zeros((0, 3)) # Has shape n_spots x 3 (x,y,z)
        self.true_spot_identities = np.zeros((0), dtype=str) # Has shape n_spots, saves every spots gene
        self.include_anchor = include_anchor
        self.include_preseq = include_preseq
        self.anchor_channel = anchor_channel
        self.seed = seed
        self.instructions = [] # Keep track of the functions called inside RoboMinnie, in order
        assert self.n_channels > 0, 'Require at least one channel'
        assert self.n_tiles > 0, 'Require at least one tile'
        assert self.n_rounds > 0, 'Require at least one round'
        assert self.n_planes > 0, 'Require at least one z plane'
        assert self.n_yx[0] > 0, 'Require y size to be at least 1 pixel'
        assert self.n_yx[1] > 0, 'Require x size to be at least 1 pixel'
        if self.include_anchor:
            assert self.anchor_channel >= 0 and self.anchor_channel < self.n_channels, \
                f'Anchor channel must be in range 0 to {self.n_channels - 1}, but got {self.anchor_channel}'
        #Ordering for data matrices is round x tile x channel x Y x X x Z, as in the nd2 file
        self.shape = (self.n_rounds, self.n_tiles, self.n_channels, self.n_yx[0], self.n_yx[1], 
            self.n_planes)
        self.anchor_shape = (self.n_tiles, self.n_channels, self.n_yx[0], self.n_yx[1], self.n_planes)
        self.preseq_shape = (self.n_rounds, self.n_tiles, self.n_channels, self.n_yx[0], self.n_yx[1], 
            self.n_planes)
        # This is the image we will build up throughout this script and eventually return. Starting with just 
        # zeroes
        self.image = np.zeros(self.shape, dtype=float)
        if self.include_anchor:
            self.anchor_image = np.zeros(self.anchor_shape, dtype=float)
        if self.include_preseq:
            self.preseq_image = np.zeros(self.shape, dtype=float)
        if self.n_tiles > 1:
            raise NotImplementedError('Multiple tile support is not implemented yet')
        if self.n_yx[0] != self.n_yx[1]:
            raise NotImplementedError('Coppafish does not support non-square tiles')
        if self.include_preseq:
            raise NotImplementedError('Preseq is not supported yet in RoboMinnie')
        if self.n_yx[0] < 2_000 or self.n_yx[1] < 2_000:
            warnings.warn('Coppafish may not support tile sizes that are too small due to implicit' + \
                ' assumptions about a tile size of 2048 in the x and y directions')
        if self.n_planes < 4:
            warnings.warn('Coppafish may break with fewer than 4 z planes')

        self.instructions.append(_funcname())
    

    def Generate_Pink_Noise(self, noise_amplitude : float, noise_spatial_scale : float) -> None:
        """
        Superimpose pink noise onto all images, including the anchor and pre-sequence images, if used. The noise 
        is identical on all images because pink noise is a good estimation for biological things that fluoresce.
        See https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.814 for more details.

        args:
            noise_amplitude (float): The maximum possible noise intensity
            noise_spatial_scale (float): Spatial scale of noise
        """
        self.instructions.append(_funcname())
        print('Generating pink noise')

        from numpy.fft import fftshift, ifftshift
        from scipy.fft import fftn, ifftn

        # True spatial scale should be maintained regardless of the image size, so we
        # scale it as such.
        self.true_noise_spatial_scale = noise_spatial_scale *  np.asarray([*self.n_yx, 10*self.n_planes])
        # Generate pink noise
        self.pink_spectrum = 1 / (
            1 + np.linspace(0, self.true_noise_spatial_scale[0], self.n_yx[0])[:,None,None]**2 +
            np.linspace(0, self.true_noise_spatial_scale[1],     self.n_yx[1])[None,:,None]**2 +
            np.linspace(0, self.true_noise_spatial_scale[2],     self.n_planes   )[None,None,:]**2
        )
        rng = np.random.RandomState(self.seed)
        self.pink_sampled_spectrum = \
            self.pink_spectrum*fftshift(fftn(rng.randn(*self.n_yx, self.n_planes)))
        self.pink_noise = np.abs(ifftn(ifftshift(self.pink_sampled_spectrum)))
        self.pink_noise = \
            (self.pink_noise - np.mean(self.pink_noise))/np.std(self.pink_noise) * noise_amplitude
        self.image[:,:,:] += self.pink_noise
        if self.include_anchor:
            self.anchor_image[:,:] += self.pink_noise


    def Generate_Random_Noise(self, noise_amplitude : float, noise_std : float, noise_mean : float = 0, 
        noise_type : str = 'normal', include_anchor : bool = True, include_preseq : bool = True) -> None:
        """
        Superimpose random, white noise onto every pixel individually.

        Args:
            noise_amplitude(float): Maximum amplitude of abs(noise)
            noise_std(float): Standard deviation/width of random noise
            noise_mean(float, optional): Mean of random noise. Default: 0
            noise_type(str('normal' or 'uniform'), optional): Type of random noise to apply. Default: 'normal'
            include_anchor(bool, optional): Whether to apply random noise to anchor round. Default: true
            include_preseq(bool, optional): Whether to apply random noise to presequence rounds. Default: true
        """
        self.instructions.append(_funcname())
        print(f'Generating random noise')

        assert noise_amplitude > 0, f'Noise amplitude must be > 0, got {noise_amplitude}'
        assert noise_std > 0, f'Noise standard deviation must be > 0, got {noise_std}'
        assert noise_mean >= 0, f'Noise mean must be >= 0, got {noise_mean}'

        # Create random pixel noise        
        rng = np.random.RandomState(self.seed)
        if noise_type == 'normal':
            noise = rng.normal(noise_mean, noise_std, size=self.shape)
            if include_anchor and self.include_anchor:
                anchor_noise = rng.normal(noise_mean, noise_std, size=self.anchor_shape)
            if include_preseq and self.include_preseq:
                preseq_noise = rng.normal(noise_mean, noise_std, size=self.preseq_shape)
        elif noise_type == 'uniform':
            noise = rng.uniform(noise_mean - noise_std/2, noise_mean+noise_std/2, size=self.shape)
            if include_anchor and self.include_anchor:
                anchor_noise = \
                    rng.uniform(noise_mean - noise_std/2, noise_mean+noise_std/2, size=self.anchor_shape)
            if include_preseq and self.include_preseq:
                preseq_noise = \
                    rng.uniform(noise_mean - noise_std/2, noise_mean+noise_std/2, size=self.preseq_shape)
        else:
            raise ValueError(f'Unknown noise type: {noise_type}')
        
        # Add new random noise to each pixel
        np.add(self.image, noise, out=self.image)
        if include_anchor and self.include_anchor:
            np.add(self.anchor_image, anchor_noise, out=self.anchor_image)
        if include_preseq and self.include_preseq:
            np.add(self.preseq_image, preseq_noise, out=self.preseq_image)


    def Add_Spots(self, n_spots : int, bleed_matrix : np.ndarray[float, float], gene_codebook_path : str, 
        spot_size_pixels : np.ndarray[float]) -> None:
        """
        Superimpose spots onto images in both space and channels (based on the bleed matrix). Also applied to the 
        anchor when included. The spots are uniformly, randomly distributed across each image.

        args:
            n_spots (int): Number of spots to add
            bleed_matrix (n_dyes x n_channels ndarray[float, float]): The bleed matrix, used to map each dye to 
            its pattern as viewed by the camera in each channel.
            gene_codebook_path (str): Path to the gene codebook, saved as a .txt file.
            spot_size_pixels (ndarray[float, float, float]): The spot's standard deviation in directions x, y, z 
            respectively.
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
        assert os.path.isfile(gene_codebook_path), f'Gene codebook at {gene_codebook_path} does not exist'
        assert spot_size_pixels.size == 3, 'Spot size must be in three dimensions'
        if bleed_matrix.shape[0] != bleed_matrix.shape[1]:
            warnings.warn(f'Given bleed matrix does not have equal channel and dye counts like usual')

        self.n_spots += n_spots

        # # Read in the gene codebook txt file
        # _codes = pandas.read_csv(gene_codebook_path, sep="\t", dtype=str).to_numpy().T
        # self.codes = {k:v for k,v in zip(*_codes)}

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
        values = list(_codes.values())
        assert len(values) == len(set(values)), f'Duplicate gene code found in dictionary: {_codes}'
        self.codes = _codes

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
        spot_img = scipy.stats.multivariate_normal([0, 0, 0], np.eye(3)*spot_size_pixels).pdf(indices.transpose(1,2,3,0))
        for p,ident in tqdm(zip(true_spot_positions_pixels, true_spot_identities), 
            desc='Superimposing spots', ascii=True, unit='spots'):

            p = np.asarray(p).astype(int)
            p_chan = np.round([self.n_channels//2, p[0], p[1], p[2]]).astype(int)
            for rnd in range(self.n_rounds):
                dye = int(self.codes[ident][rnd])
                # The second index on image is for one tile
                blit(spot_img[None,:] * bleed_matrix[dye][:,None,None,None], self.image[rnd,0], p_chan)
            blit(spot_img, self.anchor_image[0,self.anchor_channel], p)

        # Append just in case spots are created multiple times
        self.true_spot_identities = np.append(self.true_spot_identities, np.array(true_spot_identities))
        self.true_spot_positions_pixels = np.append(self.true_spot_positions_pixels, true_spot_positions_pixels, axis=0)


    # Post-Processing function
    def Fix_Image_Minimum(self, minimum : float = 0.) -> None:
        """
        Ensure that all pixels in the images are greater than given value (minimum).

        args:
            minimum (float, optional): Minimum pixel value allowed. Default: 0
        """
        self.instructions.append(_funcname())
        print(f'Fixing image minima')
        minval = np.min(self.image)
        self.image -= minval + minimum
        self.anchor_image -= minval + minimum
    

    # Post-Processing function
    def Offset_Images_By(self, constant : float, include_anchor : bool = True, include_preseq : bool = True) \
        -> None:
        """
        Shift every image pixel, in all tiles, by a constant value

        args:
            constant(float): Shift value
            include_anchor(bool, optional): Include anchor images in the shift if exists. Default: true
            include_preseq(bool, optional): Include preseq images in the shift if exists. Default: true
        """
        self.instructions.append(_funcname())
        print(f'Shifting image by {constant}')
        np.add(self.image, constant, out=self.image)
        if self.include_anchor and include_anchor:
            np.add(self.anchor_image, constant, out=self.anchor_image)
        if self.include_preseq and include_preseq:
            np.add(self.preseq_image, constant, out=self.preseq_image)


    def Save_Coppafish(self, output_dir : str) -> None:
        """
        Save known spot positions and codes, raw .npy image files, metadata.json file and config.ini file for 
        coppafish pipeline run. Output directory must be empty.
    
        args:
            output_dir(str): Save directory
        """
        self.instructions.append(_funcname())
        print(f'Saving as coppafish data')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        assert len(os.listdir(output_dir)) == 0, f'Output directory {output_dir} must be empty'

        # Create an output_dir/output_coppafish directory for coppafish pipeline output saved to disk
        self.coppafish_output = os.path.join(output_dir, 'output_coppafish')
        os.mkdir(self.coppafish_output)

        # Create an output_dir/output_coppafish/tiles directory for coppafish extract output
        self.coppafish_tiles = os.path.join(self.coppafish_output, 'tiles')
        os.mkdir(self.coppafish_tiles)

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

        #TODO: Update metadata entries for multiple tile support
        metadata = {
            "n_tiles": self.n_tiles, "n_rounds": self.n_rounds+self.include_anchor, "n_channels": self.n_channels, 
            "tile_sz": self.n_yx[0], "pixel_size_xy": 0.26, 
            "pixel_size_z": 0.9, "tile_centre": [self.n_yx[0]//2,self.n_yx[1]//2,self.n_planes//2], "tilepos_yx": [[0,0]], "tilepos_yx_nd2": [[0,0]], 
            "channel_camera": [1,1,2,3,2,3,3], "channel_laser": [1,1,2,3,2,3,3], "xy_pos": [[0,0]]
        }
        self.metadata_filepath = os.path.join(output_dir, 'metadata.json')
        with open(self.metadata_filepath, 'w') as f:
            json.dump(metadata, f)

        # Save the raw .npy files, one round at a time, in separate round directories
        self.shape
        all_images = self.image
        if self.include_anchor:
            all_images = np.concatenate([self.image, self.anchor_image[None]]).astype(np.float32)
        #TODO: Save preseq image
        for r in range(self.n_rounds + self.include_anchor):
            save_path = os.path.join(output_dir, f'{r}')
            os.mkdir(save_path)
            image_dask = dask.array.from_array(all_images[r], chunks=(self.shape[1:]))
            dask.array.to_npy_stack(save_path, image_dask)

        # Save the config.ini file
        # Finally, write the config file. z_subvols is moved from the default of 5 based on n_planes. z_box is set to
        # n_planes if the default of 12 is too large for the number of z planes
        config_file_contents = f"""; This config file is auto-generated, do not edit directly
        [file_names]
        notebook_name = notebook.npz
        input_dir = {output_dir}
        output_dir = {self.coppafish_output}
        tile_dir = {self.coppafish_tiles}
        round = {', '.join([str(i) for i in range(self.n_rounds)])}
        anchor = {self.n_rounds}
        code_book = codebook.txt
        raw_extension = .npy
        raw_metadata = {self.metadata_filepath}

        [basic_info]
        is_3d = True
        anchor_channel = {self.anchor_channel}
        dye_names = {', '.join([str(i) for i in range(self.n_channels)])}
        use_z = {', '.join([str(i) for i in range(self.n_planes)])}
        anchor_round = {self.n_rounds}
        use_rounds = {', '.join([str(i) for i in range(self.n_rounds)])}

        [register]
        z_subvols = 1
        x_subvols = 8
        y_subvols = 8
        z_box = {self.n_planes}
        """
        # Remove any large spaces in the config contents
        config_file_contents = config_file_contents.replace('  ', '')

        self.config_filepath = os.path.join(output_dir, 'config.ini')
        with open(self.config_filepath, 'w') as f:
            f.write(config_file_contents)

        # Save the instructions run so far (every function call)
        self.instruction_filepath = os.path.join(output_dir, 'instructions.txt')
        with open(self.instruction_filepath, 'w') as f:
            f.writelines(self.instructions)


    def Save(self, output_dir : str, filename : str = None) -> None:
        """
        Save RoboMinnie instance using the amazing tool pickle inside output_dir directory.

        args:
            output_dir(str): 
            filename(str, optional): Name of the pickle RoboMinnie object. Default: 'robominnie.pkl'
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
            f'RoboMinnie instance already saved as {self.instance_filepath}'

        with open(instance_filepath, 'wb') as f:
            pickle.dump(self, f)


    def Load(self, input_dir : str, filename : str = None, overwrite_self : bool = True):
        """
        Load RoboMinnie instance using the handy pickled information saved inside input_dir.

        args:
            input_dir(str): The directory where the RoboMinnie data is stored
            filename(str, optional): Name of the pickle RoboMinnie object. Default: 'robominnie.pkl'
            overwrite_self(bool, optional): If true, become the RoboMinnie instance loaded from disk

        Returns:
            RoboMinnie class loaded
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

        with open(instance_filepath, 'rb') as f:
            instance : RoboMinnie = pickle.load(f)
        
        if overwrite_self:
            self = instance
        return instance


    def Run_Coppafish(self, time_pipeline : bool = True, jax_profile_omp : bool = False) -> None:
        """
        Run RoboMinnie instance on the entire coppafish pipeline.

        args:
            time_pipeline(bool, optional): If true, print the time taken to run the coppafish pipeline. Default: \
                true
            jax_profile_omp(bool, optional): If true, profile coppafish OMP using the jax tensorboard profiler. \
                Likely requires > 32GB of RAM and more CPU time to run. Default: false
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
        run_reference_spots(nb, overwrite_ref_spots=False)
        if jax_profile_omp:
            jax.profiler.start_trace(self.coppafish_output, create_perfetto_link=True, create_perfetto_trace=True)
        run_omp(nb)
        if jax_profile_omp:
            jax.profiler.stop_trace()
        if time_pipeline:
            end_time = time.time()
            print(f'Coppafish pipeline run: {round(end_time - start_time, 1)}s. {round((end_time - start_time)//(self.n_planes * self.n_tiles), 1)}s per z plane.')

        # Keep the stitch information to convert local tiles into global coordinates
        self.tile_origins = nb.stitch.tile_origin
        # Keep the OMP spot intensities, assigned gene, assigned tile number and the spot positions in the class instance
        self.omp_spot_intensities = nb.omp.intensity
        self.omp_gene_numbers = nb.omp.gene_no
        self.omp_tile_number = nb.omp.tile
        self.omp_spot_local_positions = nb.omp.local_yxz # yxz position of each gene found
        assert self.omp_gene_numbers.shape[0] == self.omp_spot_local_positions.shape[0], 'Mismatch in spot count in ' + \
            'omp.gene_numbers and omp.local_positions'
        self.omp_spot_count = self.omp_gene_numbers.shape[0]

        if self.omp_spot_count == 0:
            warnings.warn('Copppafish OMP found zero spots')


    def Compare_Spots_OMP(self, omp_intensity_threshold : float = 0.2, location_threshold : float = 2) \
        -> Tuple[int,int,int,int]:
        """
        Compare spot positions and gene codes from coppafish OMP results to the known spot locations. If the spots 
        are close enough and the true spot has not been already assigned to an omp spot, then they are considered 
        the same spot in both coppafish output and synthetic data. If two or more spots are close enough to a true 
        spot, then the closest one is chosen. If equidistant, then take the spot with the correct gene code. If 
        not applicable, then just take the spot with the lowest index (effectively choose one of them at random, 
        but in a reproducible way).

        args:
            omp_intensity_threshold(float, optional): OMP intensity threshold, any spots below this intensity are \
                ignored. Default: 0.2
            location_threshold(float, optional): Max distance two spots can be apart to be considered the same 
            spot in pixels, inclusive. Default: 2

        returns:
            Tuple(true_positives : int, wrong_positives : int, false_positives : int, false_negatives : int): The 
            number of spots assigned to true positive, wrong positive, false positive and false negative 
            respectively, where a wrong positive is a spot assigned to the wrong gene, but found in the location 
            of a spot
        """
        self.instructions.append(_funcname())
        print(f'Comparing OMP to known spot locations')

        assert omp_intensity_threshold >= 0 and omp_intensity_threshold <= 1, \
            f'Intensity threshold must be between 0 and 1, got {omp_intensity_threshold}'
        assert location_threshold >= 0, f'Location threshold must be >= 0, got {location_threshold}'

        location_threshold_squared = location_threshold ** 2

        true_positives  = 0
        wrong_positives = 0
        false_positives = 0
        false_negatives = 0

        # Eliminate OMP spots below threshold
        omp_gene_numbers = self.omp_gene_numbers
        omp_spot_positions_yxz = self.omp_spot_local_positions
        indices = self.omp_spot_intensities > omp_intensity_threshold
        omp_gene_numbers = omp_gene_numbers[indices]
        omp_spot_positions_yxz = omp_spot_positions_yxz[indices]
        del indices

        #! Note: self.true_spot_positions_pixels has the form yxz and omp_spot_positions has the form yxz

        # TODO: Optimise this and remove the for loop through all coppafish found spots in OMP
        omp_spot_count = omp_spot_positions_yxz.shape[0]
        true_spot_count = self.true_spot_positions_pixels.shape[0]
        # Indices of every true spot index that has been paired to an omp result spot already
        true_spots_paired = np.empty(0, dtype=int)
        for s in trange(omp_spot_count, ascii=True, desc='Checking OMP spots', unit='spots'):
            omp_x = omp_spot_positions_yxz[s,1]
            omp_y = omp_spot_positions_yxz[s,0]
            omp_z = omp_spot_positions_yxz[s,2]
            omp_position_s = np.repeat([[omp_y, omp_x, omp_z]], true_spot_count, axis=0)
            # Subtract the omp position along all true spots
            position_delta = np.subtract(self.true_spot_positions_pixels, omp_position_s)
            position_delta_squared = \
                position_delta[:,0] * position_delta[:,0] + \
                position_delta[:,1] * position_delta[:,1] + \
                position_delta[:,2] * position_delta[:,2]
            
            # Find true spots close enough and closest to the OMP spot, stored as a boolean array of spots
            matches = np.logical_and(position_delta_squared <= location_threshold_squared, \
                position_delta_squared == np.min(position_delta_squared))
            
            # True spot indices
            matches_indices = np.where(matches)
            delete_indices = []
            # Ignore true spots close enough to the OMP spot if already paired to a previous OMP spot
            for i in range(len(matches_indices)):
                if matches_indices[i] in true_spots_paired:
                    delete_indices.append(i)
                    matches[matches_indices[i]] = False
            delete_indices = np.array(delete_indices, dtype=int)
            if delete_indices.size > 0:
                matches_indices = np.delete(matches_indices, delete_indices)
            matches_count = np.sum(matches)

            if matches_count == 0:
                # This omp spot is considered a false positive because there are no true spots close enough to it 
                # that have not already been paired
                false_positives += 1
                continue
            if matches_count == 1:
                # Found a single true spot matching the omp spot, see if they are the same gene (true positive)

                # Get the OMP spot gene name from the gene number. For some reason coppafish adds a new gene 
                # called Bcl11b, hence the -1
                omp_gene_name = \
                    str(list(self.codes.keys())[omp_gene_numbers[s]])
                # Actual true spot gene name as a string
                true_gene_name = str(self.true_spot_identities[matches][0])
                matching_gene = omp_gene_name == true_gene_name
                true_positives  += matching_gene
                wrong_positives += not matching_gene
                true_spots_paired = np.append(true_spots_paired, matches_indices[0])
                continue
            
            # Logic for dealing with multiple, equidistant true spots near the omp spot
            for match_index in matches_indices:
                omp_gene_name = \
                    str(list(self.codes.keys())[omp_gene_numbers[s]])
                true_gene_name = str(self.true_spot_identities[matches[match_index][0]]) # Actual true spot gene names as strings
                matching_gene = omp_gene_name == true_gene_name
                if matching_gene:
                    true_positives += 1
                    true_spots_paired = np.append(true_spots_paired, match_index)
                    continue
            # If reaching here, all close true spots are not the OMP gene
            # Assign the first true spot in the array as the pair and label it a wrong_positive
            true_spots_paired = np.append(true_spots_paired, matches_indices[0])
            wrong_positives += 1
            continue
        # False negatives are any true spots that have not been paired to an OMP spot
        false_negatives = true_spot_count - true_spots_paired.size
        return (true_positives, wrong_positives, false_positives, false_negatives)
