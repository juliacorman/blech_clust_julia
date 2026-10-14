#!/usr/bin/env python3
"""Step 1 of the LFP pipeline: validate parsed LFP channels.

Usage (run from the blech_clust directory, in the blech_clust environment):
    python validate_LFP_data.py --dataset /path/to/dataset
    python validate_LFP_data.py --data-list /path/to/list_of_datasets.txt

A dataset directory must already contain parsed LFPs in its HDF5 file
(/Parsed_LFP, created by blech_clust; they are extracted on first use if
missing), the .info file and the .params file. Any number of tastes and
regions is supported: tastes are matched to LFP arrays by dig-in name, and
regions come from the electrode layout in the .info file.

For each dataset this script:
  - assigns every parsed LFP channel to one implanted region,
  - scores channels (finite values, flat samples, robust outliers, blech_clust
    MAD artifact trials, deviation from other channels in the same region),
  - runs review-only taste-level RMS and baseline-normalized power checks,
  - writes a text report, CSV, and JSON into <data_dir>/LFP_channel_check,
  - saves voltage plots into the same directory: per-channel plots in a
    subfolder per region, plus per-region channel-check figures (all approved
    channels by taste, and the median channel by taste).

The quality thresholds are heuristics and should be reviewed for each
recording setup. validate_dataset is also used by the later pipeline steps
(remove_lfp_artifacts.py, plot_LFP_spectrogram.py, grand_average_LFP.py) to
find the approved channels of each region, so its signature should stay
stable.
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lfp_common import (
    DEFAULT_DATA_LIST,
    compute_stft,
    ephys_data,
    get_dataset_dirs,
    get_lfp_epoch_timing,
    get_taste_names_in_lfp_order,
    lfp_processing,
    normalize_trial_power,
)

# Channel-quality thresholds. These are deliberately conservative,
# scale-independent checks.
# Largest fraction of consecutive identical samples (a stuck signal).
MAX_FLAT_FRACTION = 0.01
# Largest fraction of samples beyond ROBUST_Z_LIMIT robust SDs of the median.
MAX_ARTIFACT_FRACTION = 0.01
ROBUST_Z_LIMIT = 10.0
# A channel this many robust SDs away from its region peers is rejected.
REGION_DEVIATION_Z_LIMIT = 3.0
# MAD threshold passed to blech_clust's trial artifact detector
# (remove_lfp_artifacts.py uses the same default).
BLECH_MAD_THRESHOLD = 3
# Baseline length for the review-only taste power check (the plots use
# PLOT_BASELINE_SECONDS from lfp_common).
PRE_STIM_BASELINE_SECONDS = 1.0
# The taste power check averages power from the start of the epoch up to this
# many seconds after stimulus delivery.
QC_POST_STIM_SECONDS = 3.0
# Review-only taste-level thresholds (nothing is excluded automatically).
POWER_DEVIATION_REVIEW_LIMIT = 2.0
RMS_RATIO_REVIEW_LIMIT = 2.0
# Post-stimulus time shown in the region channel check plots, in ms.
POST_STIM_PLOT_MS = 5000
# Y-axis half-range of the region channel check plots, in standard deviations
# (plot scaling only; does not affect channel validation).
Y_LIM_SD = 5


def json_default(value):
    """Convert NumPy values to types supported by the JSON encoder."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def format_number(value, spec):
    """Format a number for a report, or 'n/a' when it is missing or NaN."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return format(value, spec)


def rejection_reason(row):
    """Why a channel row was not approved, for console messages."""
    if row["quality_status"] != "good":
        return row["quality_status"]
    return (f"region {row['region_status']} (electrode is in no region or "
            f"in several regions of the .info electrode_layout)")


def rms_voltage(values):
    """Root-mean-square of the finite values of an array."""
    finite = np.isfinite(values)
    if not finite.all():
        values = values[finite]
    return float(np.sqrt(np.mean(values ** 2)))


def get_region_assignments(dat):
    """Map parsed-LFP positions to exactly one region when possible.

    The region of a channel is looked up from its electrode number in the
    .info file's electrode_layout (regions made of CAR groups of electrodes);
    parsed LFP positions index the rows of the /Parsed_LFP arrays, and
    /Parsed_LFP_channels gives the electrode of each position. A channel whose
    electrode is in no region, or in several, is reported as invalid.
    """
    dat.get_info_dict()
    dat.get_lfp_channels()
    dat.get_region_electrodes()

    assignments = []
    for lfp_position, electrode in enumerate(dat.parsed_lfp_channels):
        # A channel's region is the one whose CAR groups contain its electrode
        matches = []
        for region, car_groups in dat.region_electrode_dict.items():
            if any(electrode in car_group for car_group in car_groups):
                matches.append(region)
        # Only exactly one match counts as an assigned region
        assignments.append({
            "lfp_position": int(lfp_position),
            "electrode": int(electrode),
            "regions": matches,
            "region": matches[0] if len(matches) == 1 else "",
            "region_status": "assigned" if len(matches) == 1 else "invalid"
        })
    return assignments


def channel_quality(signal):
    """Calculate scale-independent quality metrics for one channel.

    The signal is flattened across trials. quality_status is the first failing
    check (non-finite, constant, too many flat samples, too many robust
    outliers) or "good". Region-relative checks are applied later in
    validate_dataset.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    finite = np.isfinite(signal)
    finite_fraction = float(np.mean(finite))
    finite_signal = signal[finite]

    # No usable samples: return early. Voltage summaries are None (shown as
    # n/a in reports) and are skipped when comparing against region peers.
    if finite_signal.size == 0:
        return {
            "mean_voltage": None,
            "median_voltage": None,
            "finite_fraction": finite_fraction,
            "voltage_standard_deviation": 0.0,
            "peak_to_peak": 0.0,
            "flat_fraction": 1.0,
            "artifact_fraction": 1.0,
            "quality_status": "bad_no_finite_data"
        }

    signal_std = float(np.std(finite_signal))
    peak_to_peak = float(np.ptp(finite_signal))
    # Fraction of consecutive samples that are identical (a stuck signal)
    difference = np.diff(finite_signal)
    flat_epsilon = max(signal_std * 1e-6, 1e-12)
    flat_fraction = float(np.mean(np.abs(difference) <= flat_epsilon)) \
        if difference.size else 1.0

    # Robust z-score from the MAD; 1.4826 scales MAD to an SD for normal data
    median = np.median(finite_signal)
    mad = np.median(np.abs(finite_signal - median))
    robust_scale = 1.4826 * mad
    if robust_scale == 0:
        artifact_fraction = 0.0
    else:
        robust_z = np.abs(finite_signal - median) / robust_scale
        artifact_fraction = float(np.mean(robust_z > ROBUST_Z_LIMIT))

    # First failing check wins; anything but "good" is excluded downstream
    quality_status = "good"
    if finite_fraction < 1.0:
        quality_status = "bad_nonfinite_values"
    elif signal_std == 0:
        quality_status = "bad_constant_signal"
    elif flat_fraction > MAX_FLAT_FRACTION:
        quality_status = "review_too_many_flat_samples"
    elif artifact_fraction > MAX_ARTIFACT_FRACTION:
        quality_status = "review_many_outliers"

    return {
        "mean_voltage": float(np.mean(finite_signal)),
        "median_voltage": float(np.median(finite_signal)),
        "voltage_standard_deviation": signal_std,
        "finite_fraction": finite_fraction,
        "peak_to_peak": peak_to_peak,
        "flat_fraction": flat_fraction,
        "artifact_fraction": artifact_fraction,
        "quality_status": quality_status
    }


def taste_power_qc(dat, taste_names, good_lfp_positions):
    """Compute a per-taste baseline-normalized power spectrum.

    Only channels marked good by the region/voltage checks are used. Within
    those channels, trials flagged as artifacts by blech_clust's MAD-based
    check are dropped per taste. Every remaining trial is then normalized to
    its own one-second pre-stimulus baseline at each frequency before
    trials/channels are averaged, and only those per-trial-normalized spectra
    are combined across tastes for comparison.
    """
    # Same STFT settings and timing as plot_LFP_spectrogram.py and
    # grand_average_LFP.py (see lfp_common), but only up to
    # QC_POST_STIM_SECONDS after stimulus delivery since this review-only
    # check does not need later times.
    timing = get_lfp_epoch_timing(dat)
    compute_stft(dat, timing, QC_POST_STIM_SECONDS)
    # Seconds relative to stimulus delivery (0 = delivery)
    relative_time = dat.time_vec - timing["pre_stim_s"]

    # Positions index the channel axis of the STFT and LFP arrays below
    good_positions = np.asarray(good_lfp_positions, dtype=int)

    taste_spectra = []
    excluded_trial_fractions = []
    for taste_index, amplitude_array in enumerate(dat.amplitude_array_list):
        amplitude_array = amplitude_array[:, good_positions, :, :]
        lfp_array = dat.lfp_array_list[taste_index][:, good_positions, :]
        # return_good_lfp_trial_inds expects (channels, trials, time); if the
        # check fails numerically, keep every trial
        try:
            good_trials = lfp_processing.return_good_lfp_trial_inds(
                lfp_array.swapaxes(0, 1), MAD_threshold=BLECH_MAD_THRESHOLD)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            good_trials = np.ones(amplitude_array.shape[0], dtype=bool)
        # Never drop every trial of a taste
        if not np.any(good_trials):
            good_trials = np.ones(amplitude_array.shape[0], dtype=bool)
        excluded_trial_fractions.append(
            float(1.0 - np.mean(good_trials)))

        normalized = normalize_trial_power(
            amplitude_array[good_trials], relative_time,
            PRE_STIM_BASELINE_SECONDS)
        # Average across trials and channels only after per-trial normalization.
        taste_spectra.append(np.mean(normalized, axis=(0, 1)))
    taste_spectra = np.asarray(taste_spectra)
    # taste_spectra is (taste, frequency, time); collapse both to one
    # broadband scalar per taste for the cross-taste comparison below.
    taste_broadband_power = np.mean(taste_spectra, axis=(-2, -1))

    # Compare each taste with the median of the other tastes
    power_qc = []
    for taste_index, broadband_power in enumerate(taste_broadband_power):
        other_power = np.delete(taste_broadband_power, taste_index)
        center = float(np.median(other_power)) if other_power.size \
            else float(broadband_power)
        deviation = float(broadband_power - center)
        review = abs(deviation) >= POWER_DEVIATION_REVIEW_LIMIT
        power_qc.append({
            "taste_index": taste_index,
            "taste_name": taste_names[taste_index],
            "freq_vec": dat.freq_vec,
            "normalized_power_spectrum": taste_spectra[taste_index],
            "mean_normalized_power": float(broadband_power),
            "other_taste_median_power": center,
            "power_deviation": deviation,
            "excluded_trial_fraction": excluded_trial_fractions[taste_index],
            "status": "REVIEW" if review else "PASS",
            "recommendation": "User review required; not automatically excluded"
            if review else "No large baseline-normalized power deviation detected"
        })
    return power_qc


def blech_artifact_quality(signal):
    """Run blech_clust's MAD-based artifact check on one channel.

    signal is (trials, time). blech_artifact_pass here only records whether the
    check ran without a numerical error; validate_dataset later overwrites it
    with the comparison against other channels in the same region.
    """
    # Wrap the single channel as (1, trials, time) for the channels-first check
    channel_trials = np.asarray(signal, dtype=float)
    try:
        good_trials = lfp_processing.return_good_lfp_trial_inds(
            channel_trials[np.newaxis, :, :], MAD_threshold=BLECH_MAD_THRESHOLD)
        bad_trial_fraction = float(1.0 - np.mean(good_trials))
        return {
            "blech_artifact_trial_fraction": bad_trial_fraction,
            "blech_artifact_pass": True
        }
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return {
            "blech_artifact_trial_fraction": 1.0,
            "blech_artifact_pass": False
        }


def write_validation_text(data_dir, dataset_result):
    """Write a compact per-channel validation record for one dataset.

    Channels with a missing value (for example no region peers to compare
    with) show n/a for that value.
    """
    output_dir = os.path.join(data_dir, "LFP_channel_check")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "LFP_channel_validation.txt")
    # Run-level taste QC first, then one PASS/FAIL block per channel
    with open(report_path, "w") as report:
        report.write(f"Dataset: {dataset_result['dataset']}\n")
        report.write(
            "Validation threshold: regional robust deviation <= "
            f"{REGION_DEVIATION_Z_LIMIT:g}\n")
        report.write(
            "Blech artifact threshold: regional robust deviation <= "
            f"{REGION_DEVIATION_Z_LIMIT:g}\n\n")
        report.write("Taste-level RMS QC (review only; no automatic taste "
                     "exclusions):\n")
        for taste in dataset_result["taste_qc"]:
            report.write(
                f"  {taste['taste_name']} (taste {taste['taste_index']:02d}): "
                f"RMS={taste['rms_voltage']:.6f}, "
                f"ratio to other-taste median="
                f"{format_number(taste['rms_ratio'], '.3f')}, "
                f"status={taste['status']}\n"
                f"    {taste['recommendation']}\n")
        report.write("\nTaste-level baseline-normalized power QC "
                     f"(each trial normalized to its own "
                     f"{PRE_STIM_BASELINE_SECONDS:g}s pre-stimulus baseline "
                     "before combining across tastes; review only):\n")
        for taste in dataset_result["taste_power_qc"]:
            report.write(
                f"  {taste['taste_name']} (taste {taste['taste_index']:02d}): "
                f"mean normalized power={taste['mean_normalized_power']:.3f}, "
                f"deviation from other-taste median={taste['power_deviation']:.3f}, "
                f"artifact trials excluded={taste['excluded_trial_fraction']:.3f}, "
                f"status={taste['status']}\n"
                f"    {taste['recommendation']}\n")
        # Each line re-derives a check from the stored metrics and thresholds
        report.write("\nChannel validation:\n")
        for row in dataset_result["channels"]:
            passed = row["use_for_analysis"]
            report.write(
                f"Electrode {row['electrode']} | LFP position "
                f"{row['lfp_position']} | Region: "
                f"{row['region'] or 'unassigned'} | "
                f"RESULT: {'PASS' if passed else 'FAIL'}\n")
            report.write(
                f"  Region assignment: {row['region_status']}\n"
                f"  Finite values: {'PASS' if row['finite_fraction'] == 1.0 else 'FAIL'}\n"
                f"  Flat samples: {'PASS' if row['flat_fraction'] <= MAX_FLAT_FRACTION else 'FAIL'} "
                f"({row['flat_fraction']:.5f})\n"
                f"  Robust outliers: {'PASS' if row['artifact_fraction'] <= MAX_ARTIFACT_FRACTION else 'FAIL'} "
                f"({row['artifact_fraction']:.5f})\n"
                f"  Blech artifact detection: "
                f"{'PASS' if row['blech_artifact_pass'] else 'FAIL'} "
                f"({row['blech_artifact_trial_fraction']:.5f} bad trials, "
                f"regional score "
                f"{format_number(row['blech_artifact_deviation'], '.3f')})\n"
                f"  Regional deviation: "
                f"{'PASS' if row['quality_status'] != 'bad_region_deviation' else 'FAIL'}\n"
                f"  Mean voltage: {format_number(row['mean_voltage'], '.6f')}\n"
                f"  Median voltage: "
                f"{format_number(row['median_voltage'], '.6f')}\n"
                f"  Voltage standard deviation: "
                f"{row['voltage_standard_deviation']:.6f}\n\n")
    print(f"  Validation text saved: {report_path}")


def plot_approved_channel_voltage(data_dir, dataset_result, output_dir):
    """Save one voltage plot per approved LFP channel, in a folder per region.

    Each plot shows the channel over all trials of all tastes concatenated
    end to end, against the mean and median of the approved channels in the
    same region. Traces are decimated to about 10000 points for plotting only.
    """
    data = ephys_data.ephys_data(data_dir)
    data.get_lfps()
    sampling_rate = get_lfp_epoch_timing(data)["fs"]

    dataset = dataset_result["dataset"]
    channels = dataset_result["channels"]
    os.makedirs(output_dir, exist_ok=True)

    for region in dataset_result["good_lfp_positions_by_region"]:
        assigned_rows = [row for row in channels if row["region"] == region
                         and row["region_status"] == "assigned"]
        good_positions = dataset_result["good_lfp_positions_by_region"][region]
        if not assigned_rows or not good_positions:
            continue

        # Region mean/median over approved channels, trials concatenated
        good_signal = np.asarray(
            [data.all_lfp_array[position].reshape(-1)
             for position in good_positions])
        region_mean = np.mean(good_signal, axis=0)
        region_median = np.median(good_signal, axis=0)

        plot_dir = os.path.join(output_dir, region)
        os.makedirs(plot_dir, exist_ok=True)
        # Decimate to about 10000 points so the figures stay fast to draw
        sample_count = good_signal.shape[1]
        plot_stride = max(1, int(np.ceil(sample_count / 10000)))
        plot_starts = np.arange(0, sample_count, plot_stride)
        # Samples to seconds
        x = plot_starts.astype(float) / sampling_rate
        plot_mean = region_mean[plot_starts]
        plot_median = region_median[plot_starts]

        for channel_position in good_positions:
            channel_row = next(row for row in assigned_rows
                               if row["lfp_position"] == channel_position)
            channel_signal = data.all_lfp_array[channel_position].reshape(-1)

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(x, channel_signal[plot_starts], color="darkorange",
                    linewidth=1.0, label="This channel")
            ax.plot(x, plot_mean, color="steelblue", linewidth=1.2,
                    label="Good-channel mean")
            ax.plot(x, plot_median, color="black", linewidth=1.2,
                    label="Good-channel median")

            electrode = channel_row["electrode"]
            ax.set_title(f"{dataset} | {region} | electrode {electrode} "
                         f"(approved LFP channel)")
            ax.set_xlabel("Concatenated trial time (s)")
            ax.set_ylabel("Voltage")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
            fig.tight_layout()

            base = os.path.join(
                plot_dir, f"{dataset}_{region}_electrode{electrode}")
            fig.savefig(f"{base}.png", dpi=200, bbox_inches="tight")
            fig.savefig(f"{base}.svg", bbox_inches="tight")
            plt.close(fig)
            print(f"  Voltage plots saved: {base}.png/.svg")


def plot_region_channel_check(data_dir, dataset_result, output_dir):
    """Save two channel-check figures per region, approved channels only.

    The first has a row per channel (trial-mean raw LFP, stacked with no gap)
    and a column per taste, in the style of the lfp_processing channel check.
    The second shows only the channel nearest the region's median voltage,
    with one panel per taste. Columns/panels are titled by taste name using
    the dig-in to taste mapping in the info file. Time is relative to taste
    delivery (red line at 0), and y-axes are mean +/- Y_LIM_SD SD. Files go
    directly in output_dir.
    """
    data = ephys_data.ephys_data(data_dir)
    data.get_lfps()
    dataset = dataset_result["dataset"]

    # data.lfp_array_list is one (trials, channels, time) array per taste, so
    # channels are indexed by LFP position. Tastes are named by dig-in name.
    taste_names = get_taste_names_in_lfp_order(data)
    taste_data_by_name = list(zip(taste_names, data.lfp_array_list))

    os.makedirs(output_dir, exist_ok=True)

    # Zero is taste delivery; trials start pre_stim_s before it.
    timing = get_lfp_epoch_timing(data)
    time_ms = (np.arange(data.lfp_array_list[0].shape[2])
               * 1000.0 / timing["fs"] - timing["pre_stim_s"] * 1000.0)
    taste_count = len(taste_data_by_name)

    for region, good_positions in (
            dataset_result["good_lfp_positions_by_region"].items()):
        if not good_positions:
            continue

        # (channels, time) trial means of the region's approved channels
        region_means = [
            np.mean(taste_data[:, good_positions, :], axis=0)
            for _, taste_data in taste_data_by_name]
        # One shared y-range per region so channels and tastes are comparable
        all_values = np.concatenate([m.ravel() for m in region_means])
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)

        fig, axes = plt.subplots(
            nrows=len(good_positions), ncols=taste_count,
            figsize=(4 * taste_count + 2, 8),
            sharex=True, sharey=True, squeeze=False)
        for col, (taste_name, _) in enumerate(taste_data_by_name):
            axes[0, col].set_title(taste_name, fontsize=13)
            for row, position in enumerate(good_positions):
                ax = axes[row, col]
                ax.plot(time_ms, region_means[col][row], color="tab:blue",
                        linewidth=1.0)
                ax.set_ylim([mean_val - Y_LIM_SD * std_val,
                             mean_val + Y_LIM_SD * std_val])
                ax.set_xlim(time_ms[0], min(time_ms[-1], POST_STIM_PLOT_MS))
                ax.set_yticks([])
                ax.axvline(0, color="r", linewidth=3)
                if col == 0:
                    label = ax.set_ylabel(f"Channel {position}")
                    label.set_rotation(0)
                    label.set_ha("right")
            axes[-1, col].set_xlabel("Time from taste delivery (ms)")

        fig.subplots_adjust(hspace=0, wspace=0.03, left=0.1)
        # Figure.supylabel is missing in the older matplotlib in this env
        fig.text(0.01, 0.5, "Voltage (uV)",
                 rotation=90, va="center", ha="center", fontsize=12)
        fig.suptitle(
            f"{region} - Channel Check: {dataset}\n"
            "Raw LFP Traces (trial mean) of approved channels",
            size=16, fontweight="bold")

        base = os.path.join(output_dir, f"{dataset}_{region}_channelcheck")
        fig.savefig(f"{base}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Region channel check saved: {base}.png")

        # Median figure: the approved channel nearest the region's median
        # voltage, as a single readable trace per taste
        good_rows = [row for row in dataset_result["channels"]
                     if row["lfp_position"] in good_positions]
        median_voltage = float(np.median(
            [row["median_voltage"] for row in good_rows]))
        median_row = min(
            good_rows,
            key=lambda row: abs(row["median_voltage"] - median_voltage))
        median_index = good_positions.index(median_row["lfp_position"])
        median_values = np.concatenate(
            [m[median_index] for m in region_means])
        median_mean = np.mean(median_values)
        median_std = np.std(median_values)

        ncols = 2
        nrows = int(np.ceil(taste_count / ncols))
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows),
            sharex=True, sharey=True, squeeze=False)
        for col, (taste_name, _) in enumerate(taste_data_by_name):
            ax = axes.flat[col]
            ax.plot(time_ms, region_means[col][median_index],
                    color="tab:blue", linewidth=1.0)
            ax.axvline(0, color="r", linewidth=2)
            ax.set_title(taste_name, fontsize=13)
            ax.set_ylim([median_mean - Y_LIM_SD * median_std,
                         median_mean + Y_LIM_SD * median_std])
            ax.set_xlim(time_ms[0], min(time_ms[-1], POST_STIM_PLOT_MS))
            ax.grid(True, alpha=0.25)
        # Hide unused panels in the two-column grid
        for ax in axes.flat[taste_count:]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel("Time from taste delivery (ms)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Voltage (uV)")

        fig.suptitle(
            f"{region} - Median Channel: {dataset}\n"
            f"Channel {median_row['lfp_position']} "
            f"(electrode {median_row['electrode']}), trial mean",
            size=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93))

        base = os.path.join(
            output_dir, f"{dataset}_{region}_median_channelcheck")
        fig.savefig(f"{base}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  Median channel check saved: {base}.png")


def validate_dataset(data_dir, include_power_qc=True):
    """Load one dataset and return channel, region and taste validation results.

    include_power_qc controls whether the STFT-based taste power QC is run;
    other scripts pass False when they only need channel/region validation,
    since the STFT calculation is expensive.

    The result is a dict with one row per parsed LFP channel ("channels"),
    per-region approved channel positions ("good_lfp_positions_by_region"),
    region voltage summaries, and the review-only taste checks.
    """
    basename = os.path.basename(os.path.normpath(data_dir))
    dat = ephys_data.ephys_data(data_dir)
    dat.get_lfps()
    # One name per LFP array, matched through the dig-in names so it is
    # right for any number of tastes.
    taste_names = get_taste_names_in_lfp_order(dat)
    assignments = get_region_assignments(dat)

    # Per-channel quality metrics; region comparisons come after this loop
    results = []
    for assignment in assignments:
        channel = assignment["lfp_position"]
        signal = dat.all_lfp_array[channel]
        metrics = channel_quality(signal)
        metrics.update(blech_artifact_quality(signal))
        results.append({
            "dataset": basename,
            "data_dir": data_dir,
            **assignment,
            **metrics
        })

    # Compare channel voltage level and variation only with channels in the
    # same implanted region. A region with one channel has no peer comparison.
    for region in sorted({row["region"] for row in results if row["region"]}):
        # Channels with no finite samples have no voltage statistics; they
        # are left out of the peer comparison and stay unapproved.
        region_rows = [row for row in results
                       if row["region"] == region
                       and row["median_voltage"] is not None]
        if not region_rows:
            continue
        # Spread (MAD-based) of level, variation and artifact rate across peers
        medians = np.asarray([row["median_voltage"] for row in region_rows])
        standard_deviations = np.asarray([
            row["voltage_standard_deviation"] for row in region_rows
        ])
        median_center = np.median(medians)
        median_mad = 1.4826 * np.median(np.abs(medians - median_center))
        std_center = np.median(standard_deviations)
        std_mad = 1.4826 * np.median(
            np.abs(standard_deviations - std_center))
        artifact_fractions = np.asarray([
            row["blech_artifact_trial_fraction"] for row in region_rows
        ])
        artifact_center = np.median(artifact_fractions)
        artifact_mad = 1.4826 * np.median(
            np.abs(artifact_fractions - artifact_center))

        # Deviation is in robust SDs from the region median; 0 if peers have no spread
        for row in region_rows:
            median_deviation = (
                abs(row["median_voltage"] - median_center) / median_mad
                if median_mad > 0 else 0.0)
            std_deviation = (
                abs(row["voltage_standard_deviation"] - std_center) / std_mad
                if std_mad > 0 else 0.0)
            artifact_deviation = (
                abs(row["blech_artifact_trial_fraction"] - artifact_center)
                / artifact_mad if artifact_mad > 0 else 0.0)
            row["region_median_deviation"] = float(median_deviation)
            row["region_standard_deviation_deviation"] = float(std_deviation)
            row["blech_artifact_deviation"] = float(artifact_deviation)
            # Replaces the per-channel flag from blech_artifact_quality
            row["blech_artifact_pass"] = (
                artifact_deviation <= REGION_DEVIATION_Z_LIMIT)
            deviates = max(median_deviation, std_deviation) \
                > REGION_DEVIATION_Z_LIMIT
            if deviates and row["quality_status"] == "good":
                row["quality_status"] = "bad_region_deviation"
            if (not row["blech_artifact_pass"]
                    and row["quality_status"] == "good"):
                row["quality_status"] = "bad_blech_artifact"
            # Approved only if every check passed and the region is unambiguous
            row["use_for_analysis"] = (
                row["quality_status"] == "good"
                and row["region_status"] == "assigned"
                and row["blech_artifact_pass"])

    # Rows that skipped the region loop above (no single assigned region, or
    # no finite data) get empty deviations and are never used for analysis.
    for row in results:
        row.setdefault("region_median_deviation", None)
        row.setdefault("region_standard_deviation_deviation", None)
        row.setdefault("blech_artifact_deviation", None)
        row.setdefault("use_for_analysis", False)

    # Region-level voltage summary over every assigned channel (not only
    # approved ones), with trials concatenated.
    region_summaries = []
    for region in sorted({row["region"] for row in results if row["region"]}):
        region_rows = [row for row in results
                       if row["region"] == region
                       and row["region_status"] == "assigned"]
        good_region_rows = [row for row in region_rows
                            if row["use_for_analysis"]]
        region_channels = [dat.all_lfp_array[row["lfp_position"]]
                           for row in region_rows]
        region_signal = (np.concatenate(region_channels).ravel()
                         if region_channels else np.asarray([]))
        region_signal = region_signal[np.isfinite(region_signal)]
        region_summaries.append({
            "region": region,
            "channel_count": len(region_rows),
            "good_channel_count": len(good_region_rows),
            "mean_voltage": float(np.mean(region_signal))
            if region_signal.size else None,
            "median_voltage": float(np.median(region_signal))
            if region_signal.size else None,
            "voltage_standard_deviation": float(np.std(region_signal))
            if region_signal.size else None
        })

    # Review-only: RMS of each taste's LFP (all channels) vs the median RMS of
    # the other tastes. Nothing is excluded automatically.
    taste_rms = np.asarray([
        rms_voltage(taste_data) for taste_data in dat.lfp_array_list
    ])
    taste_qc = []
    for taste_index, rms in enumerate(taste_rms):
        other_rms = np.delete(taste_rms, taste_index)
        center = float(np.median(other_rms)) if other_rms.size else float(rms)
        deviation_ratio = float(rms / center) if center else None
        taste_qc.append({
            "taste_index": taste_index,
            "taste_name": taste_names[taste_index],
            "rms_voltage": float(rms),
            "other_taste_median_rms": center,
            "rms_ratio": deviation_ratio,
            "status": "REVIEW" if deviation_ratio is not None
            and deviation_ratio >= RMS_RATIO_REVIEW_LIMIT else "PASS",
            "recommendation": "User review required; not automatically excluded"
            if deviation_ratio is not None and deviation_ratio >= RMS_RATIO_REVIEW_LIMIT
            else "No large RMS deviation detected"
        })

    # Power QC uses approved channels only and is skipped when there are none
    good_lfp_positions = [row["lfp_position"] for row in results
                         if row["use_for_analysis"]]
    power_qc = taste_power_qc(dat, taste_names, good_lfp_positions) \
        if include_power_qc and good_lfp_positions else []

    return {
        "dataset": basename,
        "data_dir": data_dir,
        "channels": results,
        "taste_qc": taste_qc,
        "taste_power_qc": power_qc,
        "region_summaries": region_summaries,
        # Approved LFP positions per region, used by the plotting functions
        "good_lfp_positions_by_region": {
            region: [row["lfp_position"] for row in results
                     if row["region"] == region and row["use_for_analysis"]]
            for region in sorted({row["region"] for row in results
                                  if row["region"]})
        }
    }


CHANNEL_CSV_FIELDS = [
    "dataset", "data_dir", "lfp_position", "electrode", "regions",
    "region", "region_status", "finite_fraction",
    "mean_voltage", "median_voltage", "voltage_standard_deviation",
    "peak_to_peak", "flat_fraction",
    "artifact_fraction", "blech_artifact_trial_fraction",
    "blech_artifact_deviation", "blech_artifact_pass",
    "region_median_deviation",
    "region_standard_deviation_deviation", "quality_status",
    "use_for_analysis"
]


def write_dataset_reports(data_dir, source, results, dataset_summary):
    """Write the channel CSV and full JSON report into the dataset's own
    LFP_channel_check directory. source records where the dataset came from
    (the data-list file or the --dataset paths)."""
    report_dir = os.path.join(data_dir, "LFP_channel_check")
    os.makedirs(report_dir, exist_ok=True)

    channel_csv = os.path.join(report_dir, "lfp_channel_validation.csv")
    with open(channel_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CHANNEL_CSV_FIELDS)
        writer.writeheader()
        for row in results:
            # Join the region list so it fits in one CSV cell
            row = dict(row)
            row["regions"] = ";".join(row["regions"])
            writer.writerow(
                {field: row.get(field, "") for field in CHANNEL_CSV_FIELDS})

    # Thresholds are stored so the report is self-describing
    report = {
        "data_list": source,
        "thresholds": {
            "max_flat_fraction": MAX_FLAT_FRACTION,
            "max_artifact_fraction": MAX_ARTIFACT_FRACTION,
            "blech_artifact_deviation_z_limit": REGION_DEVIATION_Z_LIMIT,
            "robust_z_limit": ROBUST_Z_LIMIT,
            "region_deviation_z_limit": REGION_DEVIATION_Z_LIMIT,
            "pre_stim_baseline_seconds": PRE_STIM_BASELINE_SECONDS,
            "power_deviation_review_limit": POWER_DEVIATION_REVIEW_LIMIT,
            "rms_ratio_review_limit": RMS_RATIO_REVIEW_LIMIT
        },
        "datasets": [dataset_summary],
        "channels": results
    }
    report_json = os.path.join(report_dir, "lfp_validation_report.json")
    with open(report_json, "w") as json_file:
        json.dump(report, json_file, indent=2, default=json_default)

    print(f"  Channel report saved to: {channel_csv}")
    print(f"  Full report saved to: {report_json}")


def main():
    """Validate the given datasets (--dataset) or those in a list file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-list", default=DEFAULT_DATA_LIST,
        help="Text file containing one dataset directory per line.")
    # --data-dir is the old name of this option, still accepted
    parser.add_argument(
        "--dataset", "--data-dir", action="append", default=[],
        help="Validate only this dataset path; may be provided multiple "
             "times. Overrides --data-list.")
    args = parser.parse_args()

    data_dirs = get_dataset_dirs(args.dataset, args.data_list)
    source = ", ".join(args.dataset) or args.data_list

    print(f"Validating {len(data_dirs)} datasets from {source}")
    failed = []
    for data_dir in data_dirs:
        print(f"\nProcessing {data_dir}")
        try:
            dataset_result = validate_dataset(data_dir)
            write_validation_text(data_dir, dataset_result)
            # Per-channel plots go in region subfolders; the region channel
            # checks go directly in LFP_channel_check.
            plot_approved_channel_voltage(
                data_dir, dataset_result,
                os.path.join(data_dir, "LFP_channel_check"))
            plot_region_channel_check(
                data_dir, dataset_result,
                os.path.join(data_dir, "LFP_channel_check"))
            # Console summary and JSON dataset summary
            results = dataset_result["channels"]
            good_count = sum(row["use_for_analysis"] for row in results)
            bad_rows = [row for row in results if not row["use_for_analysis"]]
            assigned_count = sum(row["region_status"] == "assigned"
                                 for row in results)
            dataset_summary = {
                "dataset": os.path.basename(os.path.normpath(data_dir)),
                "data_dir": data_dir,
                "channel_count": len(results),
                "good_channel_count": good_count,
                "assigned_region_count": assigned_count,
                "invalid_region_count": len(results) - assigned_count,
                "taste_qc": dataset_result["taste_qc"],
                "taste_power_qc": dataset_result["taste_power_qc"],
                "dataset_status": "good" if good_count == len(results)
                and assigned_count == len(results) else "review"
            }
            write_dataset_reports(
                data_dir, source, results, dataset_summary)
            print(f"  Channels: {len(results)}; good: {good_count}; "
                  f"valid region assignments: {assigned_count}")
            for summary in dataset_result["region_summaries"]:
                print(f"  {summary['region']}: "
                      f"mean={format_number(summary['mean_voltage'], '.3f')}, "
                      f"median={format_number(summary['median_voltage'], '.3f')}, "
                      f"std={format_number(summary['voltage_standard_deviation'], '.3f')} "
                      f"({summary['good_channel_count']}/"
                      f"{summary['channel_count']} good channels)")
            if bad_rows:
                print("  BAD CHANNELS:")
                for row in bad_rows:
                    print(f"    electrode {row['electrode']} "
                          f"({row['region'] or 'unknown'}): "
                          f"{rejection_reason(row)}")
            else:
                print("  No bad channels detected.")
            for taste in dataset_result["taste_qc"]:
                print(f"  Taste {taste['taste_name']}: RMS="
                      f"{taste['rms_voltage']:.3f}, ratio="
                      f"{format_number(taste['rms_ratio'], '.2f')} "
                      f"-> {taste['status']}")
            for taste in dataset_result["taste_power_qc"]:
                print(f"  Taste {taste['taste_name']}: baseline-normalized "
                      f"power={taste['mean_normalized_power']:.3f}, "
                      f"deviation={taste['power_deviation']:.2f}, "
                      f"artifact trials excluded="
                      f"{taste['excluded_trial_fraction']:.2f} -> {taste['status']}")
        # One failing dataset should not stop the rest of the list
        except Exception as error:
            print(f"  ERROR: {error}")
            failed.append(data_dir)

    if failed:
        print(f"\n{len(failed)} dataset(s) failed:")
        for data_dir in failed:
            print(f"  {data_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
