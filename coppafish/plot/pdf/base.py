import os
import textwrap
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Union, Optional, Tuple, Any

from ...setup import Notebook, NotebookPage


A4_SIZE_INCHES = (11.693, 8.268)
LARGE_FONTSIZE = 25
NORMAL_FONTSIZE = 18
SMALL_FONTSIZE = 15
SMALLER_FONTSIZE = 10
TINY_FONTSIZE = 4
INFO_FONTDICT = {"fontsize": NORMAL_FONTSIZE, "verticalalignment": "center"}


class BuildPDF:
    def __init__(self, nb: Union[Notebook, str], output_path: Optional[str] = None, overwrite: bool = True) -> None:
        """
        Build a diagnostic PDF of coppafish results.

        Args:
            nb (Notebook or str): notebook or file path to notebook.
            output_path (str, optional): file path for pdf. Default: `nb.basic_info.file_names/diagnostics.pdf`.
            overwrite (bool, optional): overwrite any other pdf outputs. Default: true.
        """
        if not overwrite:
            assert not os.path.isfile(output_path), f"overwrite is set to false but PDF already exists at {output_path}"

        if isinstance(nb, str):
            nb = Notebook(nb)

        if output_path is None:
            output_path = os.path.join(nb.file_names.output_dir, "diagnostics.pdf")
        output_path = os.path.abspath(output_path)

        mpl.rcParams.update(mpl.rcParamsDefault)
        with PdfPages(os.path.abspath(output_path)) as pdf:
            pbar = tqdm(desc="Creating Diagnostic PDF", ascii=True, total=8, unit="section")
            # Build a pdf with data from scale, extract, filter, find_spots, register, stitch, OMP
            pbar.set_postfix_str("Basic info")
            text_intro_info = self.get_basic_info(nb.basic_info, nb.file_names)
            fig, axes = self.create_empty_page(1, 1)
            self.empty_plot_ticks(axes)
            axes[0, 0].set_title(text_intro_info, fontdict=INFO_FONTDICT, y=0.5)
            pdf.savefig(fig)
            plt.close()
            pbar.update()

            pbar.set_postfix_str("Scale")
            if nb.has_page("scale"):
                text_scale_info = "Scale\n"
                text_scale_info += self.get_version_from_page(nb.scale)
                text_scale_info += f"computed scale: {nb.scale.scale}\n"
                text_scale_info += f"computed anchor scale: {nb.scale.scale_anchor}\n"
            plt.figure(figsize=A4_SIZE_INCHES, frameon=False)
            fig, axes = self.create_empty_page(1, 1)
            self.empty_plot_ticks(axes)
            axes[0, 0].set_title(text_scale_info, fontdict=INFO_FONTDICT, y=0.5)
            # Saves the current figure onto a new pdf page
            pdf.savefig()
            plt.close()
            pbar.update()

            # Extract section
            pbar.set_postfix_str("Extract")
            fig, axes = self.create_empty_page(1, 1)
            text_extract_info = ""
            text_extract_info += self.get_extract_info(nb.extract, nb.extract_debug)
            axes[0, 0].set_title(text_extract_info, fontdict=INFO_FONTDICT, y=0.5)
            extract_image_dtype = np.uint16
            self.empty_plot_ticks(axes[0, 0])
            pdf.savefig(fig)
            plt.close()
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
                    bin_size=2**9,
                )
                for fig in figs:
                    pdf.savefig(fig)
                plt.close()
                del figs
            del extract_pixel_unique_values, extract_pixel_unique_counts
            pbar.update()

            # Filter section
            pbar.set_postfix_str("Filter")
            fig, axes = self.create_empty_page(1, 1)
            text_filter_info = ""
            text_filter_info += self.get_extract_info(nb.extract, nb.extract_debug)
            axes[0, 0].set_title(text_filter_info, fontdict=INFO_FONTDICT, y=0.5)
            extract_image_dtype = np.uint16
            self.empty_plot_ticks(axes[0, 0])
            pdf.savefig(fig)
            plt.close()
            del fig, axes
            #TODO: Add filter info here
            
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
                    bin_size=2**9,
                )
                for fig in figs:
                    pdf.savefig(fig)
                plt.close()
                del figs
            del filter_pixel_unique_values, filter_pixel_unique_counts
            pbar.update()

            pbar.set_postfix_str("Find spots")
            pbar.update()

            pbar.set_postfix_str("Register")
            pbar.update()

            pbar.set_postfix_str("Stitch")
            pbar.update()

            pbar.set_postfix_str("OMP")
            pbar.update()
            pbar.close()

    def create_empty_page(
        self, nrows: int, ncols: int, hide_frames: bool = True, size: Tuple[float, float] = A4_SIZE_INCHES
    ) -> Tuple[plt.figure, np.ndarray]:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
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
                int(time_taken // 60**2), int(time_taken // 60 - time_taken // 60**2)
            )
        except AttributeError:
            time_taken = ""
        return time_taken

    def get_basic_info(self, basic_info_page: NotebookPage, file_names_page: NotebookPage) -> str:
        output = f"Coppafish {basic_info_page.software_version} Diagnostics\n"
        output += "do not edit directly, this is automatically created\n \n"
        output += "\n".join(textwrap.wrap(f"tiles: {basic_info_page.use_tiles}", 88)) + "\n"
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
        output += f"version hash: {basic_info_page.revision_hash}\n"
        input_dir = f"input directory: {file_names_page.input_dir}"
        output_dir = f"output directory: {file_names_page.output_dir}"
        wrapped_input = "...\n  ".join(textwrap.wrap(input_dir, 85))
        wrapped_output = "...\n  ".join(textwrap.wrap(output_dir, 85))
        output += f"{wrapped_input}\n"
        output += f"{wrapped_output}\n"
        return output

    def get_extract_info(self, extract_page: NotebookPage, extract_debug_page: NotebookPage) -> str:
        output = "Extract\n"
        output += self.get_version_from_page(extract_page)
        time_taken = self.get_time_taken_from_page(extract_debug_page)
        output += time_taken
        return output

    def get_filter_info(self, filter_page: NotebookPage, filter_debug_page: Optional[NotebookPage] = None) -> str:
        output = "Filter\n"
        output += self.get_version_from_page(filter_page)
        if filter_debug_page is not None:
            time_taken = self.get_time_taken_from_page(filter_debug_page)
            output += time_taken
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
        use_channels_anchor = [c for c in [nb.basic_info.dapi_channel, nb.basic_info.anchor_channel] if c is not None]
        use_channels_anchor.sort()
        use_channels = nb.basic_info.use_channels.copy()
        if nb.basic_info.dapi_channel is not None:
            use_channels += [nb.basic_info.dapi_channel]
            use_channels.sort()
        use_channels_all = use_channels_anchor.copy() + use_channels.copy()
        use_channels_all = list(set(use_channels_all))
        use_channels_all.sort()
        first_channel = use_channels_all[0]
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
                ncols=len(use_channels_all),
                size=(A4_SIZE_INCHES[0] * 2, A4_SIZE_INCHES[1] * 2),
            )
            fig.set_layout_engine("constrained")
            fig.suptitle(
                f"{section_name} pixel values{' log y axis' if log else ''}, {t=}",
                fontsize=SMALL_FONTSIZE,
            )
            maximum_count = -1
            for i, r in enumerate(use_rounds_all):
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
                    if nb.basic_info.use_anchor and r == nb.basic_info.anchor_round and c not in use_channels_anchor:
                        self.empty_plot_ticks(ax)
                        continue
                    elif r != nb.basic_info.anchor_round and c not in use_channels:
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
                    maximum_count = max([maximum_count] + hist_x)
                    ax.hist(hist_x, bins=len(hist_x), range=(pixel_min, pixel_max), log=log, color="red")
                    ax.set_xlim(pixel_min, pixel_max)
                    self.empty_plot_ticks(ax, show_bottom_frame=True)
            for i in range(axes.shape[0]):
                for j in range(axes.shape[1]):
                    axes[i, j].set_ylim(top=maximum_count)
                    if j == 0:
                        axes[i, 0].set_yticks(
                            [0, maximum_count],
                            labels=[
                                "0",
                                f"{str(maximum_count)[0]}.{str(maximum_count)[1]}e{len(str(maximum_count)) - 1}",
                            ],
                        )
            figures.append(fig)
        return figures

    def get_find_spots_info(self, find_spots_page: NotebookPage) -> str:
        pass
