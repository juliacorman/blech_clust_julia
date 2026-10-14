#!/usr/bin/env python3
"""Step 4 of the LFP pipeline: region-specific grand averages across sessions.

Usage (run from the blech_clust directory, after validate_LFP_data.py and
remove_lfp_artifacts.py):
    python grand_average_LFP.py --data-list /path/to/list_of_datasets.txt
    python grand_average_LFP.py --dataset /path/a --dataset /path/b \\
        --output-dir /path/to/results

For every dataset the cleaned LFP trials are loaded (tastes excluded in
remove_lfp_artifacts.py are skipped), the STFT is computed in memory, and the
channels approved by validate_LFP_data are grouped by region. Each trial's
power is z-scored to its own pre-stimulus baseline, and each session then
contributes one value per region and taste: the median spectrogram over
trials and channels, and the mean baseline / post-stimulus power per
frequency. Sessions are combined by averaging those per-session values, so
every session counts equally whatever its number of trials or channels.

Tastes are combined across sessions by taste NAME (as written in each .info
file), so the same taste must be spelled the same way in every session; any
number of tastes can be present, and a taste missing from some sessions is
averaged over the sessions that have it (the session count is shown in each
panel title). Sessions whose STFT time/frequency axes differ from the first
session's cannot be averaged and are skipped with a message.

Outputs, in --output-dir, for each region:
  grand_average_<region>_power_spectrum.png/.svg  pre vs post power, averaged
                                                  over tastes and sessions
  grand_average_<region>_spectrogram.png/.svg     one spectrogram per taste
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
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
from validate_LFP_data import validate_dataset

# Regions named differently in different sessions can be merged here:
# {name found in the .info file (any capitalisation): name to use}. Leave it
# empty if every session already uses the same region names.
REGION_ALIASES = {"rf1": "rf"}


def get_approved_regions(data_dir):
    """Return {region: LFP positions approved by validation} for a dataset.

    Region names are mapped through REGION_ALIASES, so positions of aliased
    regions are pooled under one name.
    """
    # Power QC (STFT) is skipped since only the region map is needed; the
    # STFT for the grand average is computed separately on cleaned trials.
    validation = validate_dataset(data_dir, include_power_qc=False)
    positions_by_region = {}
    aliases = {name.casefold(): target for name, target in REGION_ALIASES.items()}
    for region, positions in validation["good_lfp_positions_by_region"].items():
        merged_name = aliases.get(region.casefold(), region)
        positions_by_region.setdefault(merged_name, []).extend(positions)
    return {
        region: np.asarray(sorted(set(positions)), dtype=int)
        for region, positions in positions_by_region.items()
    }


def collect_session(data_dir, region_data, reference):
    """Add one session's per-region, per-taste summaries to region_data.

    region_data maps (region, taste name) to lists with one entry per
    session. reference holds the frequency and time axes of the first
    session processed, which every later session must match.
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    print(f"\nProcessing session: {dataset}")

    # Step 1: Load the epoched LFPs, then swap in the cleaned trials (this
    # drops tastes excluded in remove_lfp_artifacts.py).
    dat = ephys_data.ephys_data(data_dir)
    dat.get_lfps()
    taste_names = get_taste_names_in_lfp_order(dat)
    timing = get_lfp_epoch_timing(dat)
    load_cleaned_lfps(dat, data_dir, taste_names)
    if not dat.lfp_array_list:
        print(f"Skipping {dataset}: all tastes are excluded")
        return

    # Step 2: STFT of the full epoch, calculated in memory only.
    compute_stft(dat, timing)
    # Seconds relative to stimulus delivery (0 = delivery)
    relative_time = dat.time_vec - timing["pre_stim_s"]

    # Step 3: Every session must share one frequency and time axis to be
    # averaged together.
    if reference["freq"] is None:
        reference["freq"] = dat.freq_vec
        reference["relative_time"] = relative_time
    elif (len(dat.freq_vec) != len(reference["freq"])
          or not np.allclose(dat.freq_vec, reference["freq"])
          or len(relative_time) != len(reference["relative_time"])
          or not np.allclose(relative_time, reference["relative_time"])):
        print(f"Skipping {dataset}: its STFT time/frequency axes differ from "
              f"the first session's (different epoch length or sampling "
              f"rate), so it cannot be averaged with the others")
        return

    # Step 4: Per region, summarize each taste. A taste name can appear more
    # than once in a session (several dig-ins for one taste); those trials
    # are pooled.
    for region, channels in get_approved_regions(data_dir).items():
        if len(channels) == 0:
            print(f"  Skipping {region}: no approved LFP channels")
            continue
        amplitudes_by_taste = {}
        for taste_name, amplitude_array in zip(
                dat.active_taste_names, dat.amplitude_array_list):
            amplitudes_by_taste.setdefault(taste_name, []).append(
                amplitude_array)

        for taste_name, amplitude_arrays in amplitudes_by_taste.items():
            # Normalize each trial to its own baseline before pooling
            normalized = np.concatenate([
                normalize_trial_power(
                    amplitude_array[:, channels, :, :], relative_time,
                    PLOT_BASELINE_SECONDS)
                for amplitude_array in amplitude_arrays], axis=0)
            summary = summarize_normalized_power(normalized, relative_time)
            entry = region_data.setdefault(
                (region, taste_name),
                {"spectrograms": [], "pre": [], "post": [], "sessions": []})
            entry["spectrograms"].append(summary["spectrogram"])
            entry["pre"].append(summary["pre_power"])
            entry["post"].append(summary["post_power"])
            entry["sessions"].append(dataset)


def save_figure(fig, base, dpi):
    """Save a figure as <base>.png and <base>.svg, then close it."""
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_region(region, region_data, reference, output_dir):
    """Save the power-spectrum and spectrogram grand averages of one region."""
    freq = reference["freq"]
    time_ms = reference["relative_time"] * 1000
    time_mask = (time_ms >= SPECTROGRAM_START_MS) & (time_ms <= SPECTROGRAM_END_MS)
    freq_mask = (freq >= 0) & (freq <= PLOT_MAX_FREQ_HZ)
    safe_region = region.replace(" ", "_")

    # Average each taste over the sessions that have it, sorted by taste name
    taste_results = []
    for (entry_region, taste_name), values in sorted(
            region_data.items(), key=lambda item: item[0][1].casefold()):
        if entry_region == region:
            taste_results.append({
                "taste": taste_name,
                "spectrogram": np.mean(values["spectrograms"], axis=0),
                "session_count": len(values["sessions"])
            })

    # Pre/post power spectra are averaged over tastes within each session
    # first, then over sessions, so sessions with more tastes don't dominate.
    per_session = {}
    for (entry_region, _), values in region_data.items():
        if entry_region != region:
            continue
        for index, session in enumerate(values["sessions"]):
            session_values = per_session.setdefault(
                session, {"pre": [], "post": []})
            session_values["pre"].append(values["pre"][index])
            session_values["post"].append(values["post"][index])
    grand_pre = np.mean([np.mean(values["pre"], axis=0)
                         for values in per_session.values()], axis=0)
    grand_post = np.mean([np.mean(values["post"], axis=0)
                          for values in per_session.values()], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    window_colors = plt.get_cmap("turbo")([0.12, 0.82])
    ax.plot(freq[freq_mask], grand_pre[freq_mask],
            label="Pre-stimulus", color=window_colors[0], linewidth=2)
    ax.plot(freq[freq_mask], grand_post[freq_mask],
            label="Post-stimulus", color=window_colors[1], linewidth=2)
    ax.set(xlabel="Frequency (Hz)", ylabel="Baseline z-score")
    ax.set_title(f"{region} Grand Average Power Spectrum")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, os.path.join(
        output_dir, f"grand_average_{safe_region}_power_spectrum"), 300)

    # One spectrogram per taste, stacked, with one shared colour scale and an
    # external colourbar.
    n_tastes = len(taste_results)
    fig, axes = plt.subplots(
        n_tastes, 1, figsize=(9, 3.5 * n_tastes), squeeze=False,
        sharex=True, sharey=True)
    all_values = np.concatenate(
        [result["spectrogram"][:, time_mask].ravel()
         for result in taste_results])
    vmin, vmax = float(np.min(all_values)), float(np.max(all_values))
    for ax, result in zip(axes[:, 0], taste_results):
        pcm = ax.pcolormesh(
            time_ms[time_mask], freq, result["spectrogram"][:, time_mask],
            shading="auto", cmap="turbo", vmin=vmin, vmax=vmax)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title(
            f"{result['taste']} ({result['session_count']} sessions)",
            fontsize=14)
        ax.set_xlabel("Time (ms)", fontsize=13)
        ax.set_ylabel("Frequency (Hz)", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
    fig.suptitle(f"{region} Grand Average Spectrograms",
                 fontsize=18, y=0.985)
    fig.subplots_adjust(right=0.88, top=0.90, hspace=0.35)
    colorbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.76])
    colorbar = fig.colorbar(
        pcm, cax=colorbar_ax, label="Baseline z-scored power")
    colorbar.ax.tick_params(labelsize=11)
    colorbar.ax.yaxis.label.set_size(12)
    save_figure(fig, os.path.join(
        output_dir, f"grand_average_{safe_region}_spectrogram"), 300)

    print(f"Saved {region} taste-collapsed power and "
          f"{n_tastes} stacked spectrograms across "
          f"{len(per_session)} sessions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-list", default=DEFAULT_DATA_LIST,
        help="Text file containing one dataset directory per line.")
    parser.add_argument(
        "--dataset", action="append", default=[],
        help="Average only this dataset path; may be provided multiple "
             "times. Overrides --data-list.")
    parser.add_argument(
        "--output-dir", default=DEFAULT_GRAND_AVERAGE_DIR,
        help="Folder the figures are saved in.")
    args = parser.parse_args()

    data_dirs = get_dataset_dirs(args.dataset, args.data_list)
    region_data = {}
    reference = {"freq": None, "relative_time": None}
    failed = []
    for data_dir in data_dirs:
        try:
            collect_session(data_dir, region_data, reference)
        # One failing dataset should not stop the rest of the list
        except Exception as error:
            print(f"  ERROR processing {data_dir}: {error}")
            failed.append(data_dir)

    # Grand-average the baseline-normalized data within each region.
    if not region_data:
        print("No session produced data for any region; nothing to average.")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)
    for region in sorted({region for region, _ in region_data}):
        plot_region(region, region_data, reference, args.output_dir)

    if failed:
        print(f"\n{len(failed)} dataset(s) failed and are not in the "
              f"averages:")
        for data_dir in failed:
            print(f"  {data_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
