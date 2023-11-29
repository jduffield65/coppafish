import numpy as np
import copy

from coppafish.setup import notebook


def test_setup_notebook_merge_extract():
    rng = np.random.RandomState(0)
    master_nbp_basic = notebook.NotebookPage("basic_info")
    master_nbp_basic.n_tiles = 2
    master_nbp_basic.n_rounds = 3
    master_nbp_basic.n_extra_rounds = 1
    master_nbp_basic.n_channels = 7
    master_nbp_basic.use_tiles = [0, 1]
    
    n_pixel_values = 15
    hist_values_01 = rng.randint(100_000, size=n_pixel_values)
    hist_values_2  = rng.randint(100_000, size=n_pixel_values)
    hist_counts = rng.randint(100_000, size=(n_pixel_values, master_nbp_basic.n_rounds, master_nbp_basic.n_channels))

    extract_page_unmergeable_file_type = notebook.NotebookPage("extract")
    extract_page_unmergeable_file_type.auto_thresh = rng.randint(
        1_000, 
        size=(
            master_nbp_basic.n_tiles, 
            master_nbp_basic.n_rounds + master_nbp_basic.n_extra_rounds, 
            master_nbp_basic.n_channels
        )
    )
    extract_page_unmergeable_file_type.hist_counts = hist_counts
    extract_page_unmergeable_file_type.bg_scale = rng.rand(
        master_nbp_basic.n_tiles, 
        master_nbp_basic.n_rounds, 
        master_nbp_basic.n_channels,
    )
    extract_page_unmergeable_hist_values = copy.deepcopy(extract_page_unmergeable_file_type)
    extract_page_unmergeable_file_type.hist_values = hist_values_01
    extract_page_unmergeable_file_type.file_type = ".zarr"
    extract_page_unmergeable_hist_values.file_type = ".npy"
    extract_page_unmergeable_hist_values.hist_values = hist_values_2

    mergeable_extract_pages = []
    for _ in range(2):
        extract_page = notebook.NotebookPage("extract")
        extract_page.auto_thresh = rng.randint(
            1_000, 
            size=(
                master_nbp_basic.n_tiles, 
                master_nbp_basic.n_rounds + master_nbp_basic.n_extra_rounds, 
                master_nbp_basic.n_channels
            )
        )
        extract_page.hist_values = hist_values_01
        extract_page.hist_counts = hist_counts
        extract_page.bg_scale = rng.rand(
            master_nbp_basic.n_tiles, 
            master_nbp_basic.n_rounds, 
            master_nbp_basic.n_channels,
        )
        extract_page.file_type = ".npy"
        mergeable_extract_pages.append(extract_page)
    
    merged_extract_01 = notebook.merge_extract(mergeable_extract_pages, master_nbp_basic)
    for i, tile in enumerate(master_nbp_basic.use_tiles):
        assert np.allclose(merged_extract_01.auto_thresh[tile], mergeable_extract_pages[i].auto_thresh[tile]), \
            f"Unexpected merged auto_thresh values for tile index {i}"
        assert np.allclose(merged_extract_01.bg_scale[tile], mergeable_extract_pages[i].bg_scale[tile]), \
            f"Unexpected merged bg_scale values for tile index {i}"
    
    try:
        notebook.merge_extract(mergeable_extract_pages + [extract_page_unmergeable_file_type], master_nbp_basic)
        # Should not reach here
        assert False, "Expected an `AssertionError` to be raised by merge_extract when file_types are mismatched"
    except AssertionError:
        pass 
    try:
        notebook.merge_extract(mergeable_extract_pages + [extract_page_unmergeable_hist_values], master_nbp_basic)
        # Should not reach here
        assert False, "Expected an `AssertionError` to be raised by merge_extract when hist_values are mismatched"
    except AssertionError:
        pass
