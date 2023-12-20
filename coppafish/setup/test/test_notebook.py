import numpy as np

from coppafish.setup import notebook


def test_merge_extract():
    master_nbp_basic = notebook.NotebookPage("basic_info")
    master_nbp_basic.n_tiles = 2
    master_nbp_basic.use_tiles = [0, 1]

    mergeable_extract_pages = []
    for _ in range(3):
        extract_page = notebook.NotebookPage("extract")
        extract_page.file_type = ".npy"
        extract_page.continuous_dapi = True
        extract_page.software_version = "0.0.0"
        extract_page.revision_hash = "111"
        mergeable_extract_pages.append(extract_page)
    unmergeable_extract_pages = []
    for i in range(4):
        extract_page = notebook.NotebookPage("extract")
        extract_page.file_type = ".npy" if i != 0 else ".zarr"
        extract_page.continuous_dapi = True if i != 1 else False
        extract_page.software_version = "0.0.0" if i != 2 else "0.1.0"
        extract_page.revision_hash = "111" if i != 3 else "222"
        unmergeable_extract_pages.append(extract_page)

    result = notebook.merge_extract(mergeable_extract_pages, master_nbp_basic)
    assert result.file_type == mergeable_extract_pages[0].file_type
    assert result.continuous_dapi == mergeable_extract_pages[0].continuous_dapi
    assert result.software_version == mergeable_extract_pages[0].software_version
    assert result.revision_hash == mergeable_extract_pages[0].revision_hash
    for unmergeable_page in unmergeable_extract_pages:
        try:
            notebook.merge_extract(mergeable_extract_pages + [unmergeable_page], master_nbp_basic)
            assert False, "Expected an AssertionError to be raised due to unequal variables in extract pages"
        except AssertionError:
            pass


def test_merge_filter():
    rng = np.random.RandomState(0)
    master_nbp_basic = notebook.NotebookPage("basic_info")
    master_nbp_basic.n_tiles = 2
    master_nbp_basic.n_rounds = 3
    master_nbp_basic.n_extra_rounds = 1
    master_nbp_basic.n_channels = 7
    master_nbp_basic.use_tiles = [0, 1]

    n_pixel_values = 15
    hist_values_01 = rng.randint(100_000, size=n_pixel_values)
    hist_counts = rng.randint(
        100_000, size=(master_nbp_basic.n_tiles, n_pixel_values, master_nbp_basic.n_rounds, master_nbp_basic.n_channels)
    )

    mergeable_filter_pages = []
    for i in range(master_nbp_basic.n_tiles):
        filter_page = notebook.NotebookPage("filter")
        filter_page.auto_thresh = rng.randint(
            1_000,
            size=(
                master_nbp_basic.n_tiles,
                master_nbp_basic.n_rounds + master_nbp_basic.n_extra_rounds,
                master_nbp_basic.n_channels,
            ),
        )
        filter_page.hist_values = hist_values_01
        filter_page.hist_counts = hist_counts[i]
        filter_page.bg_scale = rng.rand(
            master_nbp_basic.n_tiles,
            master_nbp_basic.n_rounds,
            master_nbp_basic.n_channels,
        )
        filter_page.software_version = "0.0.0"
        filter_page.revision_hash = "111"
        mergeable_filter_pages.append(filter_page)
    merged_filter_01 = notebook.merge_filter(mergeable_filter_pages, master_nbp_basic)

    assert np.allclose(
        merged_filter_01.hist_counts, np.sum(hist_counts, axis=0)
    ), "hist_counts should be the sum over all tiles"
    assert merged_filter_01.software_version == mergeable_filter_pages[0].software_version, \
        "Expected same software_version after merging"
    assert merged_filter_01.revision_hash == mergeable_filter_pages[0].revision_hash, \
        "Expected same revision_hash after merging"
    for i, tile in enumerate(master_nbp_basic.use_tiles):
        assert np.allclose(
            merged_filter_01.auto_thresh[tile], mergeable_filter_pages[i].auto_thresh[tile]
        ), f"Unexpected merged auto_thresh values for tile index {i}"
        assert np.allclose(
            merged_filter_01.bg_scale[tile], mergeable_filter_pages[i].bg_scale[tile]
        ), f"Unexpected merged bg_scale values for tile index {i}"
