from ..stitch.diagnostics import shift_info_plot


def view_reg_shift_info(nb):
    shift_info = {}
    if nb.basic_info.is_3d:
        ndim = 3
    else:
        ndim = 2
    for r in nb.basic_info.use_rounds:
        name = f'Round {r}'
        shift_info[name] = {}
        shift_info[name]['shift'] = nb.register_initial_debug.shift[nb.basic_info.use_tiles, r, :ndim]
        shift_info[name]['tile'] = nb.basic_info.use_tiles
        shift_info[name]['score'] = nb.register_initial_debug.shift_score[nb.basic_info.use_tiles, r]
        shift_info[name]['score_thresh'] = nb.register_initial_debug.shift_score_thresh[nb.basic_info.use_tiles, r]
    shift_info_plot(shift_info, f"Shifts from round {nb.basic_info.ref_round}, channel {nb.basic_info.ref_channel} "
                                f"to channel {nb.register_initial_debug.shift_channel} for each round and tile")
