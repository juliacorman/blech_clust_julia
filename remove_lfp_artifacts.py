#!/usr/bin/env python3
"""Step 2 of the LFP pipeline: save artifact-free copies of the LFP trials.

Usage (run from the blech_clust directory, after validate_LFP_data.py):
    python remove_lfp_artifacts.py --dataset /path/to/dataset
    python remove_lfp_artifacts.py --data-list /path/to/list_of_datasets.txt
    python remove_lfp_artifacts.py --dataset /path/to/dataset \\
        --exclude-taste DATASET_FOLDER_NAME:Quinine

For every taste (one LFP array per taste dig-in, in dig-in name order),
trials flagged by blech_clust's MAD-based trial detector are dropped, using
only the channels approved by validate_LFP_data. A trial is dropped if it is
an outlier on any approved channel. The cleaned arrays and an audit report are
written to <data_dir>/cleaned_LFP_artifacts; the source HDF5 and data files
are never modified. plot_LFP_spectrogram.py and grand_average_LFP.py read
these outputs.

Tastes can be left out of the later plots with --exclude-taste DATASET:TASTE,
where DATASET is the dataset folder name and TASTE is the taste name as written
in the .info file (e.g. Quinine; matched exactly first, then ignoring case).
If several dig-ins share that taste name, all of them are excluded. TASTE may
instead be the taste's index (starting at 0) in the taste list printed for each
dataset. An excluded taste is still cleaned and saved; it is only recorded as
excluded in the report, and later steps skip it.
"""

import argparse
import json
import os
import sys

import numpy as np

from lfp_common import (
    CLEANED_LFP_FOLDER,
    DEFAULT_DATA_LIST,
    ephys_data,
    get_dataset_dirs,
    get_taste_names_in_lfp_order,
    lfp_processing,
)
from validate_LFP_data import validate_dataset


def resolve_taste_exclusions(specs, taste_names):
    """Turn taste names (or indices) into a set of taste indices.

    Each spec is first compared with the taste names (exactly, then ignoring
    case); only if no name matches is it read as an index, optionally written
    taste_NN like the cleaned file names. Unknown specs raise a ValueError
    listing the dataset's tastes, so a typo is never silently ignored.
    """
    names = [str(name) for name in taste_names]
    indices = set()
    for spec in specs:
        text = str(spec).strip()
        matches = ([i for i, name in enumerate(names) if name == text]
                   or [i for i, name in enumerate(names)
                       if name.casefold() == text.casefold()])
        if not matches:
            try:
                index = int(text.replace("taste_", ""))
            except ValueError:
                index = -1
            if 0 <= index < len(names):
                matches = [index]
        if not matches:
            raise ValueError(
                f"Cannot exclude taste {text!r}: it is neither a taste name "
                f"nor a valid index for this dataset's tastes "
                f"{dict(enumerate(names))}")
        indices.update(matches)
    return indices


def remove_dataset_artifacts(data_dir, mad_threshold,
                             excluded_tastes=None):
    """Write cleaned per-taste LFP arrays and an audit report for one dataset.

    excluded_tastes (taste names, or indices) are recorded in the report so
    plotting skips those tastes; they are still cleaned and saved. Returns the
    report dict.
    """
    # Step 1: Load the original epoched LFP data without modifying it.
    dat = ephys_data.ephys_data(data_dir)
    dat.get_lfps()
    taste_names = get_taste_names_in_lfp_order(dat)
    print(f"  Tastes (index: name): {dict(enumerate(taste_names))}")
    excluded_taste_indices = resolve_taste_exclusions(
        excluded_tastes or [], taste_names)
    if excluded_taste_indices:
        print("  Excluding from later plots: "
              f"{[taste_names[i] for i in sorted(excluded_taste_indices)]}")
    dataset = os.path.basename(os.path.normpath(data_dir))
    # Keep the cleaned copy beside the source dataset, never in the source
    # HDF5/data files themselves.
    output_dir = os.path.join(data_dir, CLEANED_LFP_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    # Step 2: Validate electrode identity and region assignment first.
    # Power QC (STFT) is skipped here since this step only needs the
    # channel/region validation to know which channels are approved.
    # (validate_dataset loads the LFPs itself, independently of dat.)
    validation = validate_dataset(data_dir, include_power_qc=False)
    approved_positions = sorted({position for positions in
                                 validation["good_lfp_positions_by_region"].values()
                                 for position in positions})
    report = {
        "dataset": dataset,
        "data_dir": data_dir,
        "approved_lfp_positions": approved_positions,
        "excluded_taste_indices": sorted(excluded_taste_indices),
        "excluded_taste_names": [taste_names[i]
                                 for i in sorted(excluded_taste_indices)],
        "tastes": []
    }

    for taste_index, lfp_data in enumerate(dat.lfp_array_list):
        # Step 3: Apply the existing blech_clust MAD-based trial rejection,
        # using only the approved channels. return_good_lfp_trial_inds
        # requires a trial to be good on every channel it's given, so
        # including known-bad/noisy channels here would reject trials based
        # on channel noise rather than genuine trial-level artifacts.
        # ephys_data stores trials x channels x samples; the blech_clust
        # artifact routine expects channels x trials x samples.
        # If validation approved no channels, fall back to all channels.
        if approved_positions:
            channel_trial_data = lfp_data[:, approved_positions, :].swapaxes(0, 1)
        else:
            channel_trial_data = lfp_data.swapaxes(0, 1)
        good_trials = lfp_processing.return_good_lfp_trial_inds(
            channel_trial_data, MAD_threshold=mad_threshold)
        # Fail safe: if every trial would be rejected, keep them all.
        if not np.any(good_trials):
            print(f"  WARNING: taste {taste_names[taste_index]} (index "
                  f"{taste_index}) has no trials passing the MAD threshold "
                  "on approved channels; keeping all trials")
            good_trials = np.ones(lfp_data.shape[0], dtype=bool)
        cleaned = lfp_data[good_trials]
        # Files are numbered by taste index, the same order as the LFP arrays
        file_stem = f"taste_{taste_index:02d}"
        output_path = os.path.join(output_dir, f"{file_stem}_cleaned.npy")
        np.save(output_path, cleaned)
        report["tastes"].append({
            "taste_index": taste_index,
            "taste_name": taste_names[taste_index],
            "input_shape": list(lfp_data.shape),
            "output_shape": list(cleaned.shape),
            "retained_trials": int(np.sum(good_trials)),
            "removed_trials": int(np.sum(~good_trials)),
            "taste_excluded_by_user": taste_index in excluded_taste_indices,
            "output_file": output_path
        })
        print(f"  {file_stem} ({taste_names[taste_index]}): retained "
              f"{np.sum(good_trials)}/{len(good_trials)} trials "
              f"-> {output_path}")

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
        text_report.write(
            "Approved LFP positions used for trial rejection: "
            f"{approved_positions or 'none approved (all channels used)'}\n")
        text_report.write("Only good trials were saved. Bad trials were not "
                          "saved.\n\n")
        if report["excluded_taste_indices"]:
            text_report.write(
                "User-requested taste exclusions: "
                f"{report['excluded_taste_names']} "
                f"(indices {report['excluded_taste_indices']})\n"
                "These tastes remain saved for audit purposes but will be "
                "skipped by plotting.\n\n")
        for taste in report["tastes"]:
            text_report.write(
                f"Taste {taste['taste_index']:02d} ({taste['taste_name']}): "
                f"retained {taste['retained_trials']}/"
                f"{taste['retained_trials'] + taste['removed_trials']} trials; "
                f"excluded {taste['removed_trials']} trials; "
                f"shape (trials, channels, samples) "
                f"{tuple(taste['input_shape'])} -> "
                f"{tuple(taste['output_shape'])}; "
                f"file: {taste['output_file']}\n")
        text_report.write(
            "\nA trial was excluded when its deviation from the typical "
            "LFP trial exceeded the selected MAD threshold.\n")
    print(f"Artifact explanation saved to: {text_report_path}")
    return report


def parse_exclusions(exclude_args):
    """Turn DATASET:TASTE strings into {dataset folder name: {taste, ...}}.

    TASTE is kept as text; it is resolved against each dataset's taste names
    in resolve_taste_exclusions once those are known.
    """
    exclusions = {}
    for value in exclude_args:
        dataset, separator, taste = value.rpartition(":")
        if not separator or not dataset or not taste.strip():
            raise SystemExit(
                f"--exclude-taste {value!r} must look like "
                f"DATASET_FOLDER_NAME:TASTE_NAME, e.g. my_session_240101:Quinine")
        exclusions.setdefault(dataset, set()).add(taste.strip())
    return exclusions


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
        "--mad-threshold", type=float, default=3.0,
        help="MAD threshold of the blech_clust trial artifact detector.")
    parser.add_argument(
        "--exclude-taste", action="append", default=[],
        help="Exclude a taste from the later plots using "
             "DATASET_FOLDER_NAME:TASTE, where TASTE is the taste name from "
             "the .info file (or its index), e.g. my_session_240101:Quinine; "
             "may be provided multiple times.")
    args = parser.parse_args()

    data_dirs = get_dataset_dirs(args.dataset, args.data_list)
    exclusions = parse_exclusions(args.exclude_taste)
    processed = {os.path.basename(os.path.normpath(data_dir))
                 for data_dir in data_dirs}
    for dataset in sorted(set(exclusions) - processed):
        print(f"WARNING: --exclude-taste names dataset {dataset!r}, which is "
              f"not among the datasets being processed; it is ignored.")

    failed = []
    for data_dir in data_dirs:
        print(f"\nProcessing {data_dir}")
        dataset = os.path.basename(os.path.normpath(data_dir))
        try:
            remove_dataset_artifacts(
                data_dir, args.mad_threshold, exclusions.get(dataset, set()))
        # One failing dataset should not stop the rest of the list
        except Exception as error:
            print(f"  ERROR: {error}")
            failed.append(data_dir)

    print(f"\nCleaned LFP files are in each dataset's "
          f"{CLEANED_LFP_FOLDER} folder.")
    if failed:
        print(f"{len(failed)} dataset(s) failed:")
        for data_dir in failed:
            print(f"  {data_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
