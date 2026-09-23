#!/usr/bin/env python3
"""Validate parsed LFP channels across all datasets in a directory list.

This script checks channel identity, region assignment, finite values, signal
variation, flat samples, and large robust outliers. The quality thresholds are
heuristics and should be reviewed for each recording setup.
"""

import argparse
import csv
import json
import os

import numpy as np

DEFAULT_DATA_LIST = "/home/julia-corman/Desktop/Personal_Data/data_dirs_LFP.txt"
DEFAULT_OUTPUT_DIR = "/home/julia-corman/Desktop/Personal_Data/LFP_validation"

# These are deliberately conservative, scale-independent checks.
MAX_FLAT_FRACTION = 0.01
MAX_ARTIFACT_FRACTION = 0.01
ROBUST_Z_LIMIT = 10.0
REGION_DEVIATION_Z_LIMIT = 3.0


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


def read_data_dirs(data_list_path):
    with open(data_list_path, "r") as data_file:
        return [line.strip() for line in data_file if line.strip()]


def get_region_assignments(dat):
    """Map parsed-LFP positions to exactly one region when possible."""
    dat.get_info_dict()
    dat.get_lfp_channels()
    dat.get_region_electrodes()

    assignments = []
    for lfp_position, electrode in enumerate(dat.parsed_lfp_channels):
        matches = []
        for region, car_groups in dat.region_electrode_dict.items():
            if any(electrode in car_group for car_group in car_groups):
                matches.append(region)
        assignments.append({
            "lfp_position": int(lfp_position),
            "electrode": int(electrode),
            "regions": matches,
            "region": matches[0] if len(matches) == 1 else "",
            "region_status": "assigned" if len(matches) == 1 else "invalid"
        })
    return assignments


def channel_quality(signal):
    """Calculate scale-independent quality metrics for one channel."""
    signal = np.asarray(signal, dtype=float).ravel()
    finite = np.isfinite(signal)
    finite_fraction = float(np.mean(finite))
    finite_signal = signal[finite]

    if finite_signal.size == 0:
        return {
            "finite_fraction": finite_fraction,
            "standard_deviation": 0.0,
            "peak_to_peak": 0.0,
            "flat_fraction": 1.0,
            "artifact_fraction": 1.0,
            "quality_status": "bad_no_finite_data"
        }

    signal_std = float(np.std(finite_signal))
    peak_to_peak = float(np.ptp(finite_signal))
    difference = np.diff(finite_signal)
    flat_epsilon = max(signal_std * 1e-6, 1e-12)
    flat_fraction = float(np.mean(np.abs(difference) <= flat_epsilon)) \
        if difference.size else 1.0

    median = np.median(finite_signal)
    mad = np.median(np.abs(finite_signal - median))
    robust_scale = 1.4826 * mad
    if robust_scale == 0:
        artifact_fraction = 0.0
    else:
        robust_z = np.abs(finite_signal - median) / robust_scale
        artifact_fraction = float(np.mean(robust_z > ROBUST_Z_LIMIT))

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
        "standard_deviation": signal_std,
        "peak_to_peak": peak_to_peak,
        "flat_fraction": flat_fraction,
        "artifact_fraction": artifact_fraction,
        "quality_status": quality_status
    }


def blech_artifact_quality(signal):
    """Run blech_clust's MAD-based artifact check on one channel."""
    try:
        from blech_clust.utils.ephys_data import lfp_processing
    except ModuleNotFoundError:
        from utils.ephys_data import lfp_processing

    channel_trials = np.asarray(signal, dtype=float)
    try:
        good_trials = lfp_processing.return_good_lfp_trial_inds(
            channel_trials[np.newaxis, :, :], MAD_threshold=3)
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
    """Write a compact per-channel validation record for one dataset."""
    output_dir = os.path.join(data_dir, "LFP_channel_check")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "LFP_channel_validation.txt")
    with open(report_path, "w") as report:
        report.write(f"Dataset: {dataset_result['dataset']}\n")
        report.write("Validation threshold: regional robust deviation <= 3\n")
        report.write("Blech artifact threshold: regional robust deviation <= 3\n\n")
        report.write("Taste-level RMS QC (review only; no automatic taste "
                     "exclusions):\n")
        for taste in dataset_result["taste_qc"]:
            report.write(
                f"  {taste['taste_name']} (taste {taste['taste_index']:02d}): "
                f"RMS={taste['rms_voltage']:.6f}, "
                f"ratio to other-taste median={taste['rms_ratio']:.3f}, "
                f"status={taste['status']}\n"
                f"    {taste['recommendation']}\n")
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
                f"regional score {row['blech_artifact_deviation']:.3f})\n"
                f"  Regional deviation: "
                f"{'PASS' if row['quality_status'] != 'bad_region_deviation' else 'FAIL'}\n"
                f"  Mean voltage: {row['mean_voltage']:.6f}\n"
                f"  Median voltage: {row['median_voltage']:.6f}\n"
                f"  Voltage standard deviation: "
                f"{row['voltage_standard_deviation']:.6f}\n\n")
    print(f"  Validation text saved: {report_path}")


def plot_voltage_validation(data_dir, dataset_result, output_dir):
    """Save one voltage diagnostic plot for each approved LFP channel."""
    import matplotlib.pyplot as plt
    try:
        import blech_clust.utils.ephys_data.ephys_data as ephys_data
    except ModuleNotFoundError:
        import utils.ephys_data.ephys_data as ephys_data

    data = ephys_data.ephys_data(data_dir)
    data.get_lfps()

    dataset = dataset_result["dataset"]
    channels = dataset_result["channels"]
    os.makedirs(output_dir, exist_ok=True)

    for region in dataset_result["good_lfp_positions_by_region"]:
        assigned_rows = [row for row in channels if row["region"] == region
                         and row["region_status"] == "assigned"]
        good_positions = dataset_result["good_lfp_positions_by_region"][region]
        if not assigned_rows or not good_positions:
            continue

        good_signal = np.asarray(
            [data.all_lfp_array[position].reshape(-1)
             for position in good_positions])
        region_mean = np.mean(good_signal, axis=0)
        region_median = np.median(good_signal, axis=0)

        plot_dir = os.path.join(output_dir, region)
        os.makedirs(plot_dir, exist_ok=True)
        sample_count = good_signal.shape[1]
        plot_stride = max(1, int(np.ceil(sample_count / 10000)))
        plot_starts = np.arange(0, sample_count, plot_stride)
        x = plot_starts.astype(float) / 1000.0
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


def validate_dataset(data_dir):
    """Load one dataset and return channel and region validation results."""
    try:
        import blech_clust.utils.ephys_data.ephys_data as ephys_data
    except ModuleNotFoundError:
        import utils.ephys_data.ephys_data as ephys_data

    basename = os.path.basename(os.path.normpath(data_dir))
    dat = ephys_data.ephys_data(data_dir)
    dat.get_lfps()
    dat.get_info_dict()
    taste_names = dat.info_dict["taste_params"]["tastes"]
    assignments = get_region_assignments(dat)

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
        region_rows = [row for row in results if row["region"] == region]
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
            row["blech_artifact_pass"] = (
                artifact_deviation <= REGION_DEVIATION_Z_LIMIT)
            deviates = max(median_deviation, std_deviation) \
                > REGION_DEVIATION_Z_LIMIT
            if deviates and row["quality_status"] == "good":
                row["quality_status"] = "bad_region_deviation"
            if (not row["blech_artifact_pass"]
                    and row["quality_status"] == "good"):
                row["quality_status"] = "bad_blech_artifact"
            row["use_for_analysis"] = (
                row["quality_status"] == "good"
                and row["region_status"] == "assigned"
                and row["blech_artifact_pass"])

    for row in results:
        row.setdefault("region_median_deviation", None)
        row.setdefault("region_standard_deviation_deviation", None)
        row.setdefault("blech_artifact_deviation", None)
        row.setdefault("blech_artifact_pass", False)
        row.setdefault("use_for_analysis", False)

    region_summaries = []
    for region in sorted({row["region"] for row in results if row["region"]}):
        region_rows = [row for row in results
                       if row["region"] == region
                       and row["region_status"] == "assigned"]
        good_region_rows = [row for row in region_rows
                            if row["use_for_analysis"]]
        region_channels = [dat.all_lfp_array[row["lfp_position"]]
                           for row in region_rows]
        region_signal = (np.concatenate(region_channels)
                         if region_channels else np.asarray([]))
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

    taste_rms = np.asarray([
        float(np.sqrt(np.mean(taste_data ** 2)))
        for taste_data in dat.lfp_array_list
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
            and deviation_ratio >= 3.0 else "PASS",
            "recommendation": "User review required; not automatically excluded"
            if deviation_ratio is not None and deviation_ratio >= 3.0
            else "No large RMS deviation detected"
        })

    return {
        "dataset": basename,
        "data_dir": data_dir,
        "channels": results,
        "taste_qc": taste_qc,
        "region_summaries": region_summaries,
        "good_lfp_positions_by_region": {
            region: [row["lfp_position"] for row in results
                     if row["region"] == region and row["use_for_analysis"]]
            for region in sorted({row["region"] for row in results
                                  if row["region"]})
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-list", default=DEFAULT_DATA_LIST,
        help="Text file containing one dataset directory per line.")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory where validation CSV and JSON files are written.")
    args = parser.parse_args()

    data_dirs = read_data_dirs(args.data_list)
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    dataset_summary = []

    print(f"Validating {len(data_dirs)} datasets from {args.data_list}")
    for data_dir in data_dirs:
        print(f"\nProcessing {data_dir}")
        try:
            dataset_result = validate_dataset(data_dir)
            write_validation_text(data_dir, dataset_result)
            plot_voltage_validation(
                data_dir, dataset_result,
                os.path.join(data_dir, "LFP_channel_check"))
            results = dataset_result["channels"]
            all_results.extend(results)
            good_count = sum(row["use_for_analysis"] for row in results)
            bad_rows = [row for row in results if not row["use_for_analysis"]]
            assigned_count = sum(row["region_status"] == "assigned"
                                 for row in results)
            dataset_summary.append({
                "dataset": os.path.basename(os.path.normpath(data_dir)),
                "data_dir": data_dir,
                "channel_count": len(results),
                "good_channel_count": good_count,
                "assigned_region_count": assigned_count,
                "invalid_region_count": len(results) - assigned_count,
                "taste_qc": dataset_result["taste_qc"],
                "dataset_status": "good" if good_count == len(results)
                and assigned_count == len(results) else "review"
            })
            print(f"  Channels: {len(results)}; good: {good_count}; "
                  f"valid region assignments: {assigned_count}")
            for summary in dataset_result["region_summaries"]:
                print(f"  {summary['region']}: mean={summary['mean_voltage']:.3f}, "
                      f"median={summary['median_voltage']:.3f}, "
                      f"std={summary['voltage_standard_deviation']:.3f} "
                      f"({summary['good_channel_count']}/"
                      f"{summary['channel_count']} good channels)")
            if bad_rows:
                print("  BAD CHANNELS:")
                for row in bad_rows:
                    print(f"    electrode {row['electrode']} "
                          f"({row['region'] or 'unknown'}): "
                          f"{row['quality_status']}")
            else:
                print("  No bad channels detected.")
            for taste in dataset_result["taste_qc"]:
                print(f"  Taste {taste['taste_name']}: RMS="
                      f"{taste['rms_voltage']:.3f}, ratio="
                      f"{taste['rms_ratio']:.2f} -> {taste['status']}")
        except Exception as error:
            dataset_summary.append({
                "dataset": os.path.basename(os.path.normpath(data_dir)),
                "data_dir": data_dir,
                "dataset_status": "error",
                "error": str(error)
            })
            print(f"  ERROR: {error}")

    channel_csv = os.path.join(args.output_dir, "lfp_channel_validation.csv")
    with open(channel_csv, "w", newline="") as csv_file:
        fieldnames = [
            "dataset", "data_dir", "lfp_position", "electrode", "regions",
            "region", "region_status", "finite_fraction",
            "mean_voltage", "median_voltage", "voltage_standard_deviation",
            "standard_deviation", "peak_to_peak", "flat_fraction",
            "artifact_fraction", "blech_artifact_trial_fraction",
            "blech_artifact_deviation", "blech_artifact_pass",
            "region_median_deviation",
            "region_standard_deviation_deviation", "quality_status",
            "use_for_analysis"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            row = dict(row)
            row["regions"] = ";".join(row["regions"])
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    report = {
        "data_list": args.data_list,
        "thresholds": {
            "max_flat_fraction": MAX_FLAT_FRACTION,
            "max_artifact_fraction": MAX_ARTIFACT_FRACTION,
            "blech_artifact_deviation_z_limit": REGION_DEVIATION_Z_LIMIT,
            "robust_z_limit": ROBUST_Z_LIMIT,
            "region_deviation_z_limit": REGION_DEVIATION_Z_LIMIT
        },
        "datasets": dataset_summary,
        "channels": all_results
    }
    report_json = os.path.join(args.output_dir, "lfp_validation_report.json")
    with open(report_json, "w") as json_file:
        json.dump(report, json_file, indent=2, default=json_default)

    print(f"\nChannel report saved to: {channel_csv}")
    print(f"Full report saved to: {report_json}")


if __name__ == "__main__":
    main()
