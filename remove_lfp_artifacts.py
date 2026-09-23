#!/usr/bin/env python3
"""Remove LFP trial artifacts into separate files without changing source data."""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')

from validate_LFP_data import validate_dataset

try:
    import blech_clust.utils.ephys_data.ephys_data as ephys_data
    from blech_clust.utils.ephys_data import lfp_processing
except ModuleNotFoundError:
    import utils.ephys_data.ephys_data as ephys_data
    from utils.ephys_data import lfp_processing


DEFAULT_DATA_LIST = "/home/julia-corman/Desktop/Personal_Data/data_dirs_LFP.txt"


def read_data_dirs(path):
    with open(path, "r") as data_file:
        return [line.strip() for line in data_file if line.strip()]


def remove_dataset_artifacts(data_dir, mad_threshold, output_root,
                             excluded_taste_indices=None):
    # Step 1: Load the original epoched LFP data without modifying it.
    dat = ephys_data.ephys_data(data_dir)
    dat.get_lfps()
    dataset = os.path.basename(os.path.normpath(data_dir))
    # Keep the cleaned copy beside the source dataset, never in the source
    # HDF5/data files themselves.
    output_dir = os.path.join(data_dir, "cleaned_LFP_artifacts")
    os.makedirs(output_dir, exist_ok=True)
    # Step 2: Validate electrode identity and region assignment first.
    validation = validate_dataset(data_dir)
    approved_positions = sorted({position for positions in
                                 validation["good_lfp_positions_by_region"].values()
                                 for position in positions})
    report = {
        "dataset": dataset,
        "data_dir": data_dir,
        "approved_lfp_positions": approved_positions,
        "excluded_taste_indices": sorted(excluded_taste_indices or []),
        "tastes": []
    }

    for taste_index, lfp_data in enumerate(dat.lfp_array_list):
        # Step 3: Apply the existing blech_clust MAD-based trial rejection.
        # Only the retained trials are written to the new dataset copy.
        # ephys_data stores trials x channels x samples; the blech_clust
        # artifact routine expects channels x trials x samples.
        channel_trial_data = lfp_data.swapaxes(0, 1)
        good_trials = lfp_processing.return_good_lfp_trial_inds(
            channel_trial_data, MAD_threshold=mad_threshold)
        cleaned = lfp_data[good_trials]
        taste_name = f"taste_{taste_index:02d}"
        output_path = os.path.join(output_dir, f"{taste_name}_cleaned.npy")
        np.save(output_path, cleaned)
        report["tastes"].append({
            "taste_index": taste_index,
            "input_shape": list(lfp_data.shape),
            "output_shape": list(cleaned.shape),
            "retained_trials": int(np.sum(good_trials)),
            "removed_trials": int(np.sum(~good_trials)),
            "taste_excluded_by_user": taste_index in (excluded_taste_indices or []),
            "channel_axis_preserved": True,
            "output_file": output_path
        })
        print(f"  {taste_name}: retained {np.sum(good_trials)}/"
              f"{len(good_trials)} trials -> {output_path}")

    # Step 4: Save an audit report describing retained and excluded trials.
    report_path = os.path.join(output_dir, "artifact_removal_report.json")
    with open(report_path, "w") as report_file:
        json.dump(report, report_file, indent=2)

    text_report_path = os.path.join(output_dir, "artifact_removal_report.txt")
    with open(text_report_path, "w") as text_report:
        text_report.write(f"Dataset: {dataset}\n")
        text_report.write("Artifact removal method: blech_clust MAD-based "
                          "trial detector\n")
        text_report.write(f"MAD threshold: {mad_threshold}\n")
        text_report.write("Only good trials were saved. Bad trials were not "
                          "saved.\n\n")
        if report["excluded_taste_indices"]:
            text_report.write(
                "User-requested taste exclusions: "
                f"{report['excluded_taste_indices']}\n"
                "These tastes remain saved for audit purposes but will be "
                "skipped by plotting.\n\n")
        for taste in report["tastes"]:
            text_report.write(
                f"Taste {taste['taste_index']:02d}: "
                f"retained {taste['retained_trials']}/"
                f"{taste['retained_trials'] + taste['removed_trials']} trials; "
                f"excluded {taste['removed_trials']} trials\n")
        text_report.write(
            "\nA trial was excluded when its deviation from the typical "
            "LFP trial exceeded the selected MAD threshold.\n")
    print(f"Artifact explanation saved to: {text_report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-list", default=DEFAULT_DATA_LIST)
    parser.add_argument(
        "--dataset", action="append", default=[],
        help="Process only this dataset path; may be provided multiple times.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Deprecated: cleaned files are now saved inside each dataset.")
    parser.add_argument("--mad-threshold", type=float, default=3.0)
    parser.add_argument(
        "--exclude-taste", action="append", default=[],
        help="Exclude a taste using DATASET:taste index, e.g. "
             "JC13_taste_day1_260722_152746:1")
    args = parser.parse_args()

    data_dirs = args.dataset or read_data_dirs(args.data_list)
    for data_dir in data_dirs:
        print(f"\nProcessing {data_dir}")
        dataset = os.path.basename(os.path.normpath(data_dir))
        exclusions = {
            int(value.split(":", 1)[1].replace("taste_", ""))
            for value in args.exclude_taste
            if value.split(":", 1)[0] == dataset
        }
        remove_dataset_artifacts(
            data_dir, args.mad_threshold, args.output_root, exclusions)
        print("\nCleaned LFP files saved inside each dataset's "
            "cleaned_LFP_artifacts folder.")


if __name__ == "__main__":
    main()
