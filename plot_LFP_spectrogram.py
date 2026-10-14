#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3 of the LFP pipeline: per-session spectrograms and power spectra.

Usage (run from the blech_clust directory, after validate_LFP_data.py and
remove_lfp_artifacts.py):
    python plot_LFP_spectrogram.py --dataset /path/to/dataset
    python plot_LFP_spectrogram.py --data-list /path/to/list_of_datasets.txt

For each dataset this script:
  1. loads the cleaned LFP trials written by remove_lfp_artifacts.py (tastes
     excluded there are skipped) and names each taste through the .info file,
  2. computes the STFT of every trial and channel in memory (nothing is written
     into the HDF5 file),
  3. keeps only the channels approved by validate_LFP_data, grouped by region,
  4. z-scores each trial's power to its own pre-stimulus baseline,
  5. saves, into <data_dir>/LFP_analyses:
       - <name>_STFT_data.pkl: the raw STFT amplitudes and axes,
       - <name>_<region>_STFT_spectrogram.png/.svg: median spectrogram with one
         panel per taste, time aligned to stimulus delivery (0 ms, red line),
       - <name>_<region>_power_spectrum_0-35Hz_by_taste.png/.svg: baseline vs
         post-stimulus power by frequency, one colour per taste.
It also writes lfp_region_validation.json (the channel validation of every
dataset) into --output-dir. Any number of tastes and regions is supported;
grand_average_LFP.py combines sessions afterwards.
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from lfp_common import (
    DEFAULT_DATA_LIST,
    DEFAULT_GRAND_AVERAGE_DIR,
    PLOT_BASELINE_SECONDS,
    PLOT_MAX_FREQ_HZ,
    SPECTROGRAM_END_MS,
    SPECTROGRAM_START_MS,
    compute_stft,
    ephys_data,
    get_dataset_dirs,
    get_lfp_epoch_timing,
    get_taste_names_in_lfp_order,
    load_cleaned_lfps,
    normalize_trial_power,
    summarize_normalized_power,
)
from validate_LFP_data import (
    format_number, json_default, rejection_reason, validate_dataset)


def validate_regions(data_dir):
    """Validate one dataset's channels and print what was approved.

    Returns the full validation result; its
    result["good_lfp_positions_by_region"] maps each region to the LFP
    positions (rows of the LFP arrays) approved for analysis. Power QC (STFT)
    is skipped since only the channel/region validation is needed here. The
    text report is written by validate_LFP_data.py, not here.
    """
    result = validate_dataset(data_dir, include_power_qc=False)

    print(f"\nRegion validation: {os.path.basename(os.path.normpath(data_dir))}")
    for summary in result["region_summaries"]:
        print(f"  {summary['region']}: "
              f"mean={format_number(summary['mean_voltage'], '.3f')}, "
              f"median={format_number(summary['median_voltage'], '.3f')}, "
              f"std={format_number(summary['voltage_standard_deviation'], '.3f')} "
              f"({summary['good_channel_count']}/{summary['channel_count']} "
              f"good channels)")
    bad_rows = [row for row in result["channels"] if not row["use_for_analysis"]]
    if bad_rows:
        print("  BAD CHANNELS:")
        for row in bad_rows:
            print(f"    electrode {row['electrode']} "
                  f"({row['region'] or 'unknown'}): {rejection_reason(row)}")
    else:
        print("  No bad channels detected.")
    return result


def summarize_region(amplitude_arrays, channel_positions, relative_time):
    """Baseline-normalize one region's power and reduce it to what is plotted.

    amplitude_arrays holds one (trials, channels, frequencies, time) array per
    taste. Each trial is z-scored to its own baseline first; then, per taste,
    the median over trials and channels gives the spectrogram, and the mean
    over trials, channels and time inside the baseline / post-stimulus window
    gives the pre and post power spectra. The large normalized arrays are
    discarded right away so memory does not grow with the number of tastes.
    """
    summaries = []
    for amplitude_array in amplitude_arrays:
        normalized = normalize_trial_power(
            amplitude_array[:, channel_positions, :, :], relative_time,
            PLOT_BASELINE_SECONDS)
        summaries.append(summarize_normalized_power(normalized, relative_time))
    return summaries


def plot_spectrograms(save_base, basename, region, channel_count,
                      taste_names, summaries, freq_vec, relative_time):
    """Save one spectrogram panel per taste with a shared colour scale."""
    time_ms = relative_time * 1000
    time_mask = (time_ms >= SPECTROGRAM_START_MS) & (time_ms <= SPECTROGRAM_END_MS)
    x_ticks = np.arange(SPECTROGRAM_START_MS, SPECTROGRAM_END_MS + 1, 250)
    y_ticks = np.arange(freq_vec.min(), freq_vec.max() + 1, 5)

    # One colour range for all tastes so they can be compared by eye. It is
    # taken over the whole epoch, not only the displayed window.
    all_values = np.concatenate(
        [summary["spectrogram"].ravel() for summary in summaries])
    vmin, vmax = np.min(all_values), np.max(all_values)

    num_tastes = len(summaries)
    fig, axes = plt.subplots(
        num_tastes, 1, figsize=(12, 8 * num_tastes), sharex=True,
        squeeze=False)
    for ax, taste_name, summary in zip(axes[:, 0], taste_names, summaries):
        pcm = ax.pcolormesh(
            time_ms[time_mask], freq_vec, summary["spectrogram"][:, time_mask],
            shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
        ax.axvline(0, color='red', linewidth=2, linestyle='--')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title(f"{taste_name} | {basename} | "
                     f"{region} channels: {channel_count}", fontsize=24)
    axes[-1, 0].set_xlabel("Time (ms)", fontsize=24)
    fig.text(0.05, 0.5, "Frequency (Hz)", va='center',
             rotation='vertical', fontsize=24)
    fig.subplots_adjust(right=0.88, hspace=0.5)
    colorbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    colorbar = fig.colorbar(
        pcm, cax=colorbar_ax, label="Baseline z-scored power")
    colorbar.ax.tick_params(labelsize=18)
    colorbar.ax.yaxis.label.set_size(18)

    fig.savefig(f'{save_base}.svg', dpi=300, bbox_inches='tight')
    fig.savefig(f'{save_base}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"{region} spectrogram figures saved to: {save_base}")


def plot_power_spectra(save_base, basename, region, taste_names, summaries,
                       freq_vec):
    """Compare baseline (dashed) and post-stimulus (solid) power per taste."""
    freq_mask = (freq_vec >= 0) & (freq_vec <= PLOT_MAX_FREQ_HZ)
    taste_colors = plt.get_cmap('turbo')(
        np.linspace(0.08, 0.92, len(summaries)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for color, taste_name, summary in zip(taste_colors, taste_names, summaries):
        ax.plot(freq_vec[freq_mask], summary["pre_power"][freq_mask],
                label=f'{taste_name} baseline', linewidth=1.8,
                color=color, linestyle='--')
        ax.plot(freq_vec[freq_mask], summary["post_power"][freq_mask],
                label=f'{taste_name} post-stimulus', linewidth=1.8,
                color=color)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Baseline z-score')
    ax.set_title(f'{basename} - {region} Taste Power Spectra '
                 f'0-{PLOT_MAX_FREQ_HZ} Hz')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(f'{save_base}.svg', dpi=300, bbox_inches='tight')
    fig.savefig(f'{save_base}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"{region} taste-comparison power plots saved: {save_base}")


def process_session(data_dir, region_map):
    """Compute the STFT for one dataset and save its plots.

    region_map maps region name -> LFP positions approved by validation
    (validate_regions(data_dir)["good_lfp_positions_by_region"]).
    """
    basename = os.path.basename(os.path.normpath(data_dir))
    print(f"\nProcessing session: {basename}")

    # Step 1: Load the epoched LFPs. Each epoch runs from before to after
    # stimulus delivery; the timing is recovered from the data itself.
    dat = ephys_data.ephys_data(data_dir)
    dat.get_lfps()
    taste_names = get_taste_names_in_lfp_order(dat)
    timing = get_lfp_epoch_timing(dat)

    # Step 2: Swap in the cleaned trials from remove_lfp_artifacts.py. This
    # leaves the original dataset unchanged and drops excluded tastes.
    load_cleaned_lfps(dat, data_dir, taste_names)
    if not dat.lfp_array_list:
        print("  All tastes are excluded; skipping STFT and plots")
        return
    taste_list = dat.active_taste_names

    # Step 3: STFT of the full epoch, calculated only for retained tastes in
    # memory. write_out=False prevents results from being written to the HDF5.
    compute_stft(dat, timing)
    print("Frequencies:", dat.freq_vec)
    # Seconds relative to stimulus delivery (0 = delivery)
    relative_time = dat.time_vec - timing["pre_stim_s"]

    save_dir = os.path.join(data_dir, 'LFP_analyses')
    os.makedirs(save_dir, exist_ok=True)

    # Step 4: Save the STFT results for later reuse.
    save_path = os.path.join(save_dir, f"{basename}_STFT_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "amplitude_array_list": dat.amplitude_array_list,
            "freq_vec": dat.freq_vec,
            "time_vec": dat.time_vec,
            "pre_stim_s": timing["pre_stim_s"],
            "amp_array_shape": [x.shape for x in dat.amplitude_array_list],
            "info_dict": dat.info_dict,
            "taste_list": taste_list,
            "taste_indices": dat.active_taste_indices
        }, f)
    print(f"All variables saved to {save_path}")

    # Step 5: One set of plots per region, using only approved channels.
    for region, channel_positions in region_map.items():
        if len(channel_positions) == 0:
            print(f"  Skipping {region}: no approved LFP channels")
            continue
        summaries = summarize_region(
            dat.amplitude_array_list, channel_positions, relative_time)
        plot_spectrograms(
            os.path.join(save_dir, f'{basename}_{region}_STFT_spectrogram'),
            basename, region, len(channel_positions), taste_list, summaries,
            dat.freq_vec, relative_time)
        plot_power_spectra(
            os.path.join(
                save_dir,
                f'{basename}_{region}_power_spectrum_'
                f'0-{PLOT_MAX_FREQ_HZ}Hz_by_taste'),
            basename, region, taste_list, summaries, dat.freq_vec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-list", default=DEFAULT_DATA_LIST,
        help="Text file containing one dataset directory per line.")
    parser.add_argument(
        "--dataset", action="append", default=[],
        help="Process only this dataset path; may be provided multiple "
             "times. Overrides --data-list.")
    parser.add_argument(
        "--output-dir", default=DEFAULT_GRAND_AVERAGE_DIR,
        help="Folder for lfp_region_validation.json.")
    args = parser.parse_args()

    # Only the requested directories are processed; this never touches the
    # source LFP data and relies on remove_lfp_artifacts.py's cleaned,
    # good-trial-only copies of every taste it loads.
    run_dirs = get_dataset_dirs(args.dataset, args.data_list)

    # Validate each dataset first (which channels belong to which region and
    # are usable), then plot it. A failing dataset is reported at the end
    # instead of stopping the run.
    validations = []
    failed = []
    for data_dir in run_dirs:
        try:
            validation = validate_regions(data_dir)
            validations.append(validation)
            process_session(
                data_dir, validation["good_lfp_positions_by_region"])
        except Exception as error:
            print(f"  ERROR processing {data_dir}: {error}")
            failed.append(data_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    validation_path = os.path.join(
        args.output_dir, 'lfp_region_validation.json')
    with open(validation_path, "w") as f:
        json.dump(validations, f, indent=2, default=json_default)
    print(f"Region validation saved to: {validation_path}")

    # Grand averages across sessions come from grand_average_LFP.py.
    if failed:
        print(f"\n{len(failed)} dataset(s) failed:")
        for data_dir in failed:
            print(f"  {data_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
