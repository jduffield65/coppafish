import os
import warnings
import textwrap
import webbrowser
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Union, Optional, Tuple, Any

from ...setup import Notebook, NotebookPage


# Plot settings
A4_SIZE_INCHES = (11.693, 8.268)
LARGE_FONTSIZE = 25
NORMAL_FONTSIZE = 18
SMALL_FONTSIZE = 15
SMALLER_FONTSIZE = 10
TINY_FONTSIZE = 4
INFO_FONTDICT = {"fontsize": NORMAL_FONTSIZE, "verticalalignment": "center"}
N_GENES_SHOW = 40


class BuildPDF:
    def __init__(
        self,
        nb: Union[Notebook, str],
        output_path: Optional[str] = None,
        overwrite: bool = True,
        auto_open: bool = True,
    ) -> None:
        """
        Build a diagnostic PDF of coppafish results.

        Args:
            nb (Notebook or str): notebook or file path to notebook.
            output_path (str, optional): file path for pdf. Default: `nb.basic_info.file_names/diagnostics.pdf`.
            overwrite (bool, optional): overwrite any other pdf outputs. Default: true.
            auto_open (bool, optional): open the PDF in a web browser after creation. Default: true.
        """
        if output_path is None:
            output_path = os.path.join(nb.file_names.output_dir, "diagnostics.pdf")
        output_path = os.path.abspath(output_path)

        if not overwrite:
            assert not os.path.isfile(output_path), f"overwrite is set to false but PDF already exists at {output_path}"

        with PdfPages(os.path.abspath(output_path)) as pdf:
            pbar = tqdm(desc="Creating Diagnostic PDF", total=10, unit="section")
            pbar.set_postfix_str("Loading notebook")

            if isinstance(nb, str):
                nb = Notebook(nb)
            self.use_channels_anchor = [
                c for c in [nb.basic_info.dapi_channel, nb.basic_info.anchor_channel] if c is not None
            ]
            self.use_channels_anchor.sort()
            self.use_channels_plus_dapi = nb.basic_info.use_channels.copy()
            if nb.basic_info.dapi_channel is not None:
                self.use_channels_plus_dapi += [nb.basic_info.dapi_channel]
            self.use_channels_all = self.use_channels_plus_dapi.copy()
            if nb.basic_info.anchor_channel is not None:
                self.use_channels_all += [nb.basic_info.anchor_channel]
            self.use_channels_all = list(set(self.use_channels_all))
            self.use_channels_all.sort()
            self.use_channels_plus_dapi.sort()
            self.use_rounds_all = (
                nb.basic_info.use_rounds.copy()
                + nb.basic_info.use_anchor * [nb.basic_info.anchor_round]
                + nb.basic_info.use_preseq * [nb.basic_info.pre_seq_round]
            )
            self.use_rounds_all.sort()
            mpl.rcParams.update(mpl.rcParamsDefault)
            pbar.update()

            # Build a pdf with data from scale, extract, filter, find_spots, register, stitch, OMP
            pbar.set_postfix_str("Basic info")
            text_intro_info = self.get_basic_info(nb.basic_info, nb.file_names)
            fig, axes = self.create_empty_page(1, 1)
            self.empty_plot_ticks(axes)
            axes[0, 0].set_title(text_intro_info, fontdict=INFO_FONTDICT, y=0.5)
            pdf.savefig(fig)
            plt.close(fig)
            pbar.update()

            pbar.set_postfix_str("Scale")
            if nb.has_page("scale"):
                text_scale_info = "Scale\n \n"
                text_scale_info += self.get_version_from_page(nb.scale)
                text_scale_info += f"computed scale: {nb.scale.scale}\n"
                text_scale_info += f"computed anchor scale: {nb.scale.scale_anchor}\n"
                plt.figure(figsize=A4_SIZE_INCHES, frameon=False)
                fig, axes = self.create_empty_page(1, 1)
                self.empty_plot_ticks(axes)
                axes[0, 0].set_title(text_scale_info, fontdict=INFO_FONTDICT, y=0.5)
                # Saves the current figure onto a new pdf page
                pdf.savefig(fig)
                plt.close(fig)
            pbar.update()

            # Extract section
            pbar.set_postfix_str("Extract")
            if nb.has_page("extract"):
                fig, axes = self.create_empty_page(1, 1)
                text_extract_info = ""
                text_extract_info += self.get_extract_info(nb.extract, nb.extract_debug)
                axes[0, 0].set_title(text_extract_info, fontdict=INFO_FONTDICT, y=0.5)
                extract_image_dtype = np.uint16
                self.empty_plot_ticks(axes[0, 0])
                pdf.savefig(fig)
                plt.close(fig)
                del fig, axes
                try:
                    extract_pixel_unique_values = nb.extract_debug.pixel_unique_values.copy()
                    extract_pixel_unique_counts = nb.extract_debug.pixel_unique_counts.copy()
                except AttributeError:
                    extract_pixel_unique_values = None
                    extract_pixel_unique_counts = None
                if extract_pixel_unique_values is not None:
                    pixel_min, pixel_max = np.iinfo(extract_image_dtype).min, np.iinfo(extract_image_dtype).max
                    # Histograms of pixel value histograms
                    figs = self.create_pixel_value_hists(
                        nb,
                        "Extract",
                        extract_pixel_unique_values,
                        extract_pixel_unique_counts,
                        pixel_min,
                        pixel_max,
                        bin_size=2**10,
                    )
                    for fig in figs:
                        pdf.savefig(fig)
                        plt.close(fig)
                    del figs
                del extract_pixel_unique_values, extract_pixel_unique_counts
            pbar.update()

            # Filter section
            pbar.set_postfix_str("Filter")
            if nb.has_page("filter") or nb.has_page("extract"):
                fig, axes = self.create_empty_page(1, 1)
                text_filter_info = ""
                if nb.has_page("filter"):
                    # Versions >=0.5.0
                    text_filter_info += self.get_filter_info(nb.filter, nb.filter_debug)
                else:
                    text_filter_info += self.get_filter_info(nb.extract, nb.extract_debug)
                axes[0, 0].set_title(text_filter_info, fontdict=INFO_FONTDICT, y=0.5)
                self.empty_plot_ticks(axes[0, 0])
                pdf.savefig(fig)
                plt.close(fig)

                filter_image_dtype = np.uint16
                try:
                    filter_pixel_unique_values = nb.filter_debug.pixel_unique_values.copy()
                    filter_pixel_unique_counts = nb.filter_debug.pixel_unique_counts.copy()
                except AttributeError:
                    filter_pixel_unique_values = None
                    filter_pixel_unique_counts = None
                if filter_pixel_unique_values is not None:
                    pixel_min, pixel_max = np.iinfo(filter_image_dtype).min, np.iinfo(filter_image_dtype).max
                    # Histograms of pixel value histograms
                    figs = self.create_pixel_value_hists(
                        nb,
                        "Filter",
                        filter_pixel_unique_values,
                        filter_pixel_unique_counts,
                        pixel_min,
                        pixel_max,
                        bin_size=2**10,
                    )
                    for fig in figs:
                        pdf.savefig(fig)
                        plt.close(fig)
                    del figs
                del filter_pixel_unique_values, filter_pixel_unique_counts
            pbar.update()

            pbar.set_postfix_str("Find spots")
            if nb.has_page("find_spots"):
                fig, axes = self.create_empty_page(1, 1)
                text_find_spots_info = ""
                text_find_spots_info += self.get_find_spots_info(nb.find_spots)
                axes[0, 0].set_title(text_find_spots_info, fontdict=INFO_FONTDICT, y=0.5)
                self.empty_plot_ticks(axes[0, 0])
                pdf.savefig(fig)
                plt.close(fig)

                minimum_spot_count = nb.find_spots.spot_no[nb.find_spots.spot_no != 0].min()
                maximum_spot_count = nb.find_spots.spot_no.max()
                for t in nb.basic_info.use_tiles:
                    fig, axes = self.create_empty_page(1, 1)
                    fig.suptitle(f"Find spot counts, tile {t}")
                    ax: plt.Axes = axes[0, 0]
                    channels_to_index = {c: i for i, c in enumerate(self.use_channels_all)}
                    X = np.zeros(
                        (nb.basic_info.n_rounds + nb.basic_info.n_extra_rounds, len(channels_to_index)), dtype=np.int32
                    )
                    ticks_channels = np.arange(X.shape[1])
                    ticks_channels_labels = ["" for _ in range(ticks_channels.size)]
                    ticks_rounds = np.arange(X.shape[0])
                    ticks_rounds_labels = ["" for _ in range(ticks_rounds.size)]
                    for r in self.use_rounds_all:
                        if nb.basic_info.use_anchor and r == nb.basic_info.anchor_round:
                            use_channels = [
                                c for c in [nb.basic_info.dapi_channel, nb.basic_info.anchor_channel] if c is not None
                            ]
                        else:
                            use_channels = nb.basic_info.use_channels.copy()
                        for c in use_channels:
                            X[r, channels_to_index[c]] = nb.find_spots.spot_no[t, r, c]
                            ticks_channels_labels[channels_to_index[c]] = f"{c}"
                            if nb.basic_info.dapi_channel is not None and c == nb.basic_info.dapi_channel:
                                ticks_channels_labels[channels_to_index[c]] = f"dapi"
                            if nb.basic_info.anchor_channel is not None and c == nb.basic_info.anchor_channel:
                                ticks_channels_labels[channels_to_index[c]] = f"anchor"
                            ticks_rounds_labels[r] = f"{r if r != nb.basic_info.anchor_round else 'anchor'}"
                            if r == nb.basic_info.pre_seq_round:
                                ticks_rounds_labels[r] = f"preseq"
                    im = ax.imshow(X, cmap="viridis", norm="log", vmin=minimum_spot_count, vmax=maximum_spot_count)
                    ax.set_xlabel("Channels")
                    ax.set_xticks(ticks_channels)
                    ax.set_xticklabels(ticks_channels_labels)
                    ax.set_yticks(ticks_rounds)
                    ax.set_yticklabels(ticks_rounds_labels)
                    ax.set_ylabel("Rounds")
                    # Create colour bar
                    cbar = ax.figure.colorbar(im, ax=ax)
                    cbar.ax.set_ylabel("Spot count", rotation=-90, va="bottom")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
            pbar.update()

            pbar.set_postfix_str("Register")
            pbar.update()

            pbar.set_postfix_str("Stitch")
            pbar.update()

            pbar.set_postfix_str("Reference and call spots")
            if nb.has_page("ref_spots") and nb.has_page("call_spots"):
                # Create a page for every gene
                gene_probs = nb.ref_spots.gene_probs
                n_genes = gene_probs.shape[1]
                gene_names = nb.call_spots.gene_names
                colours = (
                    nb.ref_spots.colors[:, :, nb.basic_info.use_channels].copy() + nb.basic_info.tile_pixel_value_shift
                )
                spot_colours_rnorm = colours / np.linalg.norm(colours, axis=2)[:, :, None]
                n_rounds = spot_colours_rnorm.shape[1]
                for g in range(n_genes):
                    g_spots = np.argsort(-gene_probs[:, g])
                    # Sorted probabilities, with greatest score at index 0
                    g_probs = gene_probs[g_spots, g]
                    g_bled_code = nb.call_spots.bled_codes[g, :, nb.basic_info.use_channels]  # rounds x channels
                    g_r_dot_products = np.sum(spot_colours_rnorm * g_bled_code[None, :, :], 2)
                    fig, axes = self.create_empty_page(
                        2,
                        2,
                        share_x=True,
                        gridspec_kw={"width_ratios": [2, 1]},
                    )
                    fig.suptitle(f"{gene_names[g]}", size=NORMAL_FONTSIZE)
                    im = axes[0, 0].imshow(g_r_dot_products[g_spots[:N_GENES_SHOW], :].T, vmin=0, vmax=1, aspect="auto")
                    axes[0, 0].set_yticks(range(n_rounds))
                    axes[0, 0].set_ylim([n_rounds - 0.5, -0.5])
                    axes[0, 0].set_ylabel("round")
                    axes[0, 0].set_title(f"dye code match")
                    self.empty_plot_ticks(axes[1, 0], show_bottom_frame=True, show_left_frame=True)
                    axes[1, 0].plot(np.arange(N_GENES_SHOW), g_probs[:N_GENES_SHOW])
                    axes[1, 0].plot(np.arange(N_GENES_SHOW), g_r_dot_products[g_spots[:N_GENES_SHOW]].mean(1))
                    axes[1, 0].legend(("probability score", "mean match"), loc="lower right")
                    axes[1, 0].set_xticks(
                        [i for i in range(N_GENES_SHOW)], labels=[str(i) for i in range(N_GENES_SHOW)]
                    )
                    axes[1, 0].set_xlabel("spot number (ranked by probability)")
                    axes[1, 0].set_ylim([0, 1])
                    axes[1, 0].set_yticks([0, 0.25, 0.5, 0.75, 1])
                    axes[1, 0].grid(True)
                    axes[0, 0].autoscale(enable=True, axis="x", tight=True)
                    axes[0, 1].imshow(g_bled_code, vmin=0, vmax=1)
                    axes[0, 1].set_title("bled code")
                    axes[0, 1].set_xlabel("channels")
                    axes[0, 1].set_xticks(np.arange(len(nb.basic_info.use_channels)), labels=nb.basic_info.use_channels)
                    cbar = fig.colorbar(im, ax=axes[0, 1], orientation="vertical")
                    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
                    self.empty_plot_ticks(axes[1, 1])
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
            pbar.update()

            pbar.set_postfix_str("OMP")
            pbar.update()
            pbar.close()

        if auto_open:
            webbrowser.open_new_tab(rf"{output_path}")

    def create_empty_page(
        self,
        nrows: int,
        ncols: int,
        hide_frames: bool = True,
        size: Tuple[float, float] = A4_SIZE_INCHES,
        share_x: bool = False,
        share_y: bool = False,
        gridspec_kw: dict = {},
    ) -> Tuple[plt.figure, np.ndarray]:
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, squeeze=False, sharex=share_x, sharey=share_y, gridspec_kw=gridspec_kw
        )
        fig.set_size_inches(size)
        for ax in axes.ravel():
            ax.set_frame_on(not hide_frames)
        fig.tight_layout()
        return fig, axes

    def empty_plot_ticks(
        self,
        axes: Union[np.ndarray, plt.Axes],
        show_top_frame: bool = False,
        show_bottom_frame: bool = False,
        show_left_frame: bool = False,
        show_right_frame: bool = False,
    ) -> None:
        def _apply_to(axis: plt.Axes):
            axis.set_frame_on(True)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines["top"].set_visible(show_top_frame)
            axis.spines["bottom"].set_visible(show_bottom_frame)
            axis.spines["left"].set_visible(show_left_frame)
            axis.spines["right"].set_visible(show_right_frame)

        if isinstance(axes, np.ndarray):
            for ax in axes.ravel():
                _apply_to(ax)
        else:
            ax = axes
            _apply_to(ax)

    def get_version_from_page(self, page: NotebookPage) -> str:
        output = ""
        try:
            output += f"version: {page.software_version}\n"
            output += f"version hash: {page.revision_hash}\n\n"
        except AttributeError:
            pass
        return output

    def get_time_taken_from_page(self, page: NotebookPage) -> str:
        try:
            time_taken = page.time_taken
            time_taken = "time taken: {0} hour(s) and {1} minute(s)\n".format(
                int(time_taken // 60**2), int((time_taken // 60) % 60)
            )
        except AttributeError:
            time_taken = ""
        return time_taken

    def get_basic_info(self, basic_info_page: NotebookPage, file_names_page: NotebookPage) -> str:
        try:
            output = f"Coppafish {basic_info_page.software_version} Diagnostics"
        except AttributeError:
            output = f"Coppafish <0.5.0 Diagnostics"
        output += "\n \n"
        use_tiles = basic_info_page.use_tiles
        output += "\n".join(textwrap.wrap(f"{len(use_tiles)} tiles: {use_tiles}", 88)) + "\n"
        output += (
            "...\n".join(
                textwrap.wrap(
                    f"{3 if basic_info_page.is_3d else 2}D, tile dimensions: "
                    + f"{basic_info_page.nz if basic_info_page.is_3d else 1}x{basic_info_page.tile_sz}x"
                    + f"{basic_info_page.tile_sz}",
                    85,
                )
            )
            + "\n"
        )
        output += f"sequencing rounds: {basic_info_page.use_rounds}\n"
        if basic_info_page.use_anchor:
            output += (
                f"anchor round: {basic_info_page.anchor_round}\nanchor channel: {basic_info_page.anchor_channel}\n"
            )
        if basic_info_page.use_preseq:
            output += f"presequence round: {basic_info_page.pre_seq_round}\n"
        output += f"channels used: {basic_info_page.use_channels}\n"
        if basic_info_page.dapi_channel is not None:
            output += f"dapi channel: {basic_info_page.dapi_channel}\n"
        try:
            output += f"version hash: {basic_info_page.revision_hash}\n"
        except AttributeError:
            pass
        input_dir = f"input directory: {file_names_page.input_dir}"
        output_dir = f"output directory: {file_names_page.output_dir}"
        wrapped_input = "...\n  ".join(textwrap.wrap(input_dir, 85))
        wrapped_output = "...\n  ".join(textwrap.wrap(output_dir, 85))
        output += f"{wrapped_input}\n"
        output += f"{wrapped_output}\n"
        return output

    def get_extract_info(self, extract_page: NotebookPage, extract_debug_page: NotebookPage) -> str:
        output = "Extract\n \n"
        output += self.get_version_from_page(extract_page)
        time_taken = self.get_time_taken_from_page(extract_debug_page)
        output += time_taken
        return output

    def get_filter_info(self, filter_page: NotebookPage, filter_debug_page: Optional[NotebookPage] = None) -> str:
        output = "Filter\n \n"
        output += self.get_version_from_page(filter_page)
        if filter_debug_page is not None:
            time_taken = self.get_time_taken_from_page(filter_debug_page)
            output += time_taken
        if filter_debug_page.r_dapi is not None:
            # Filtering DAPI is true
            output += f"dapi filtering with r_dapi: {filter_debug_page.r_dapi}"
        else:
            output += f"no dapi filtering"
        return output

    def create_pixel_value_hists(
        self,
        nb: Notebook,
        section_name: str,
        pixel_unique_values: np.ndarray,
        pixel_unique_counts: np.ndarray,
        pixel_min: int,
        pixel_max: int,
        bin_size: int,
        log: bool = True,
    ) -> list:
        assert bin_size >= 1
        assert (pixel_max - pixel_min + 1) % bin_size == 0

        figures = []
        use_channels = nb.basic_info.use_channels.copy()
        if nb.basic_info.dapi_channel is not None:
            use_channels += [nb.basic_info.dapi_channel]
            use_channels.sort()
        use_channels_all = list(set(use_channels + self.use_channels_anchor))
        use_channels_all.sort()
        first_channel = use_channels[0]
        use_rounds_all = nb.basic_info.use_rounds.copy()
        if nb.basic_info.use_anchor:
            use_rounds_all += [nb.basic_info.anchor_round]
        if nb.basic_info.use_preseq:
            use_rounds_all += [nb.basic_info.pre_seq_round]
        use_rounds_all = list(set(use_rounds_all))
        use_rounds_all.sort()
        final_round = use_rounds_all[-1]
        for t in nb.basic_info.use_tiles:
            fig, axes = self.create_empty_page(
                nrows=len(use_rounds_all),
                ncols=len(set(use_channels + self.use_channels_anchor)),
                size=(A4_SIZE_INCHES[0] * 2, A4_SIZE_INCHES[1] * 2),
                share_y=True,
            )
            fig.set_layout_engine("constrained")
            fig.suptitle(
                f"{section_name} pixel values{' log y axis' if log else ''}, {t=}",
                fontsize=SMALL_FONTSIZE,
            )
            for i, r in enumerate(use_rounds_all):
                if r == nb.basic_info.anchor_round:
                    use_channels_r = self.use_channels_anchor
                else:
                    use_channels_r = nb.basic_info.use_channels.copy()
                    if nb.basic_info.dapi_channel is not None:
                        use_channels_r += [nb.basic_info.dapi_channel]
                for j, c in enumerate(use_channels_all):
                    ax: plt.Axes = axes[i, j]
                    if c == first_channel:
                        round_label = r
                        if nb.basic_info.use_anchor and r == nb.basic_info.anchor_round:
                            round_label = "anchor"
                        elif nb.basic_info.use_preseq and r == nb.basic_info.pre_seq_round:
                            round_label = "preseq"
                        ax.set_ylabel(
                            f"round {round_label}",
                            fontdict={"fontsize": SMALL_FONTSIZE},
                        )
                    if r == final_round:
                        ax.set_xlabel(
                            f"channel {c if c != nb.basic_info.dapi_channel else 'dapi'}",
                            fontdict={"fontsize": SMALL_FONTSIZE},
                        )
                    self.empty_plot_ticks(ax, show_bottom_frame=True)
                    if c not in use_channels_r:
                        self.empty_plot_ticks(ax)
                        continue
                    hist_x = []
                    k = 0
                    for pixel_value in range(pixel_max + 1):
                        if pixel_value == pixel_unique_values[t, r, c, k]:
                            hist_x.append(pixel_unique_counts[t, r, c, k])
                            k += 1
                        else:
                            hist_x.append(0)
                    if bin_size > 1:
                        new_hist_x = [0 for _ in range(len(hist_x) // bin_size)]
                        for k in range(len(hist_x) // bin_size):
                            for l in range(bin_size):
                                new_hist_x[k] += hist_x[k * bin_size + l]
                        hist_x = new_hist_x
                    if np.sum(hist_x) <= 0:
                        warnings.warn(f"The {section_name.lower()} image for {t=}, {r=}, {c=} looks to be all zeroes!")
                        continue
                    ax.hist(hist_x, bins=len(hist_x), range=(pixel_min, pixel_max), log=log, color="red")
                    ax.set_xlim(pixel_min, pixel_max)
                    self.empty_plot_ticks(ax, show_bottom_frame=True)
            # for i in range(axes.shape[0]):
            #     for j in range(axes.shape[1]):
            #         if j == 0:
            #             axes[i, 0].set_yticks(minor=False)
            figures.append(fig)
        return figures

    def get_find_spots_info(self, find_spots_page: NotebookPage) -> str:
        output = "Find Spots\n \n"
        output += self.get_version_from_page(find_spots_page)
        time_taken = self.get_time_taken_from_page(find_spots_page)
        output += time_taken
        return output
