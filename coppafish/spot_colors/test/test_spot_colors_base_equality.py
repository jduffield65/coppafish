import os
import pytest
import numpy as np

from coppafish.utils import tiles_io


@pytest.mark.optimised
def test_get_spot_colors_equality():
    from coppafish.spot_colors.base import get_spot_colors
    from coppafish.spot_colors.base_optimised import get_spot_colors as get_spot_colors_jax

    rng = np.random.RandomState(13)

    # We require disk written extracted tiles to test this function 
    directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_test_dir')
    if not os.path.isdir(directory):
        os.mkdir(directory)

    n_tiles = 1
    n_rounds = 2
    n_channels = 3
    tile_sz = 15
    n_planes = 3

    for file_type in ['.npy', '.zarr']:
        nbp_file_dict = dict()
        nbp_file_dict_tile = list(np.zeros((n_tiles, n_rounds + 1, n_channels), str))

        for c in range(n_channels):
            for r in range(n_rounds):
                # zyx saving format
                image = rng.rand(n_planes, tile_sz, tile_sz)
                filepath = os.path.join(directory, f'r{r}c{c}{file_type}')
                nbp_file_dict_tile[0][r][c] = filepath
                tiles_io.save_image(image, filepath, file_type)
            # Save a preseq image
            preseq_image = rng.rand(n_planes, tile_sz, tile_sz)
            filepath = os.path.join(directory, f'preseq_c{c}{file_type}')
            nbp_file_dict_tile[0][n_rounds][c] = filepath
            tiles_io.save_image(preseq_image, filepath, file_type)
        nbp_file_dict['tile'] = nbp_file_dict_tile
        #TODO: 2d testing
        for is_3d in [True]:
            for use_bg in [True, False]:
                nbp_basic_dict = dict()
                nbp_basic_dict['use_preseq'] = use_bg
                nbp_basic_dict['use_rounds'] = list(np.arange(n_rounds))
                nbp_basic_dict['use_channels'] = list(np.arange(n_channels))
                nbp_basic_dict['is_3d'] = is_3d
                nbp_basic_dict['tile_sz'] = tile_sz
                nbp_basic_dict['use_z'] = list(np.arange(n_planes))
                nbp_basic_dict['pre_seq_round'] = n_rounds
                nbp_basic_dict['tile_pixel_value_shift'] = 1


@pytest.mark.optimised
def test_apply_transform_equality():
    from coppafish.spot_colors.base import apply_transform
    from coppafish.spot_colors.base_optimised import apply_transform as apply_transform_jax
    
    rng = np.random.RandomState(9)
    n_spots = 100
    tile_sz = np.asarray([15, 16, 17], dtype=np.int16)
    yxz = rng.randint(15, size=(n_spots, 3), dtype=np.int16)
    transform = rng.rand(4, 3) * 2
    yxz_transform, in_range = apply_transform(yxz, transform, tile_sz)
    assert yxz_transform.shape == (n_spots, 3)
    assert in_range.shape == (n_spots, )
    yxz_transform_jax, in_range_jax = apply_transform_jax(yxz, transform, tile_sz)
    assert yxz_transform_jax.shape == (n_spots, 3)
    assert in_range_jax.shape == (n_spots, )
    assert np.allclose(yxz_transform, yxz_transform_jax)
    assert np.all(in_range == in_range_jax)


@pytest.mark.optimised
def test_all_pixel_yxz_equality():
    from coppafish.spot_colors.base import all_pixel_yxz
    from coppafish.spot_colors.base_optimised import all_pixel_yxz as all_pixel_yxz_jax

    y_size, x_size = 11, 12
    z_planes_options = [13, [13], np.asarray([4,5]), [2,3,4]]
    n_z_planes = [1, 1, 2, 3]
    for i, z_planes in enumerate(z_planes_options):
        output = all_pixel_yxz(y_size, x_size, z_planes)
        assert output.shape == (y_size * x_size * n_z_planes[i], 3)
        assert output.dtype == np.int16
        output_jax = all_pixel_yxz_jax(y_size, x_size, z_planes)
        assert output_jax.shape == (y_size * x_size * n_z_planes[i], 3)
        assert output_jax.dtype == np.int16
        assert np.all(output == output_jax)
