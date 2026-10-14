"""Helpers shared by the LFP validation / cleaning / plotting scripts.

The pipeline is run in this order from the blech_clust directory:

  1. validate_LFP_data.py     - decide which channels are usable, per region
  2. remove_lfp_artifacts.py  - save artifact-free copies of the LFP trials
  3. plot_LFP_spectrogram.py  - per-session spectrograms and power spectra
  4. grand_average_LFP.py     - averages across sessions

All four scripts need the same answers to a few questions, and answering them
differently in different scripts would silently misalign the data. They live
here so there is exactly one implementation:

  * How long is each LFP epoch, and where is stimulus delivery inside it?
    (get_lfp_epoch_timing)
  * Which taste does each parsed LFP array belong to?
    (get_taste_names_in_lfp_order)
  * Which STFT settings do we use? (make_stft_params)
  * How is power normalized to the pre-stimulus baseline?
    (normalize_trial_power)
  * How are the cleaned trials loaded back in? (load_cleaned_lfps)

Where the data come from (nothing here reads trial_info_frame.csv directly):
  * <data_dir>/*.h5      /Parsed_LFP/<dig_in_name>_LFPs  - one LFP array per
                         taste dig-in, already cut around stimulus delivery
                         (blech_clust did this from trial_info_frame.csv when
                         the LFPs were first extracted), plus
                         /Parsed_LFP_channels - the electrode of each LFP row.
  * <data_dir>/*.info    taste names and dig-in names (taste_params) and the
                         electrode-to-region layout (electrode_layout).
  * <data_dir>/*.params  spike_array_durations [pre, post] in ms, the first
                         choice for where stimulus delivery falls in each
                         LFP epoch (checked against the array length).
"""

import json
import os

import numpy as np
import tables

# The blech_clust modules are imported here once and re-exported, so the
# scripts import them from lfp_common. Inside the repository directory the
# package may only be importable as "utils".
try:
    import blech_clust.utils.ephys_data.ephys_data as ephys_data
    from blech_clust.utils.ephys_data import lfp_processing
except ModuleNotFoundError:
    import utils.ephys_data.ephys_data as ephys_data  # noqa: F401
    from utils.ephys_data import lfp_processing  # noqa: F401

# Defaults for the command-line options of every script. Override them with
# --data-list / --output-dir; edit them here to change the defaults for all
# scripts at once.
DEFAULT_DATA_LIST = "/home/julia-corman/Desktop/Personal_Data/data_dirs_LFP.txt"
DEFAULT_GRAND_AVERAGE_DIR = (
    "/home/julia-corman/Desktop/Personal_Data/grand_average_results")

# Folder (inside each data directory) written by remove_lfp_artifacts.py
CLEANED_LFP_FOLDER = "cleaned_LFP_artifacts"

# The pipeline assumes LFPs were extracted at this sampling rate (the
# blech_clust default); a warning is printed if the extraction settings say
# otherwise.
LFP_SAMPLING_RATE_HZ = 1000

# STFT settings. The window is defined in seconds (500 samples at 1 kHz).
STFT_WINDOW_SECONDS = 0.5
STFT_MAX_FREQ_HZ = 40
# The STFT window smears power across the stimulus boundary, so baselines end
# this many seconds before stimulus delivery (half the window length).
BASELINE_END_BUFFER_SECONDS = STFT_WINDOW_SECONDS / 2

# Windows used by the plots (seconds relative to stimulus delivery, at 0).
# The baseline starts this far before stimulus delivery; if a recording has a
# shorter pre-stimulus period the baseline is simply what is available.
PLOT_BASELINE_SECONDS = 2.0
# "Post-stimulus" power is averaged from 0 to this many seconds.
PLOT_POST_STIM_SECONDS = 1.75
# Power-spectrum plots show frequencies up to this value (Hz).
PLOT_MAX_FREQ_HZ = 35
# Spectrograms show this window around stimulus delivery, in ms.
SPECTROGRAM_START_MS = -1000
SPECTROGRAM_END_MS = 1750


def read_data_dirs(data_list_path):
    """Return the non-empty, stripped lines of a text file as directory paths."""
    with open(data_list_path, "r") as data_file:
        return [line.strip() for line in data_file if line.strip()]


def get_dataset_dirs(data_dir_args, data_list_path):
    """Dataset directories to process: explicit ones if given, else the list."""
    return list(data_dir_args) or read_data_dirs(data_list_path)


def get_lfp_epoch_timing(dat):
    """Return the timing of the parsed LFP epochs of one dataset.

    dat.get_lfps() must have been called. Each LFP trial is a window around
    stimulus delivery: pre_stim_s seconds before it and post_stim_s seconds
    after it, sampled at fs Hz. The window lengths are not stored with the
    data, so they are recovered by finding the configuration whose total
    length matches the number of samples actually in the arrays. Candidates,
    in order of preference: spike_array_durations from the .params file (the
    one setting shared by every dataset), the LFP parameters set on the
    ephys_data object, then the blech_clust default LFP parameters (used when
    LFPs are extracted with no overrides). The first one that fits the arrays
    is used, and a warning is printed if another fitting candidate puts
    stimulus delivery at a different time. If none fit, stopping is safer than
    plotting data with the wrong time zero.

    The sampling rate is taken to be LFP_SAMPLING_RATE_HZ; a warning is
    printed if the extraction settings give a different rate.

    Returns a dict with pre_stim_s, post_stim_s, total_s and fs.
    """
    n_samples = dat.lfp_array_list[0].shape[-1]
    fs = LFP_SAMPLING_RATE_HZ
    extracted_fs = (dat.lfp_params.get("fin_sampling_rate")
                    or dat.default_lfp_params["fin_sampling_rate"])
    if extracted_fs != fs:
        print(f"  WARNING: the LFP extraction settings give a sampling rate "
              f"of {extracted_fs} Hz, but this pipeline assumes {fs} Hz. "
              f"Frequencies and times in the plots will be wrong unless the "
              f"stored LFPs really are at {fs} Hz.")

    candidates = []
    try:
        dat.get_sorting_params_dict()
        candidates.append((
            ".params spike_array_durations",
            dat.sorting_params_dict["spike_array_durations"]))
    except (IndexError, OSError, KeyError, ValueError):
        pass  # no usable .params file; the other candidates may still match
    candidates += [
        ("ephys_data.lfp_params['trial_durations']",
         dat.lfp_params.get("trial_durations")),
        ("default LFP trial_durations",
         dat.default_lfp_params["trial_durations"]),
    ]

    fitting = [
        (name, durations) for name, durations in candidates
        if durations is not None and len(durations) == 2
        and int(round(sum(durations) * fs / 1000.0)) == n_samples]
    if fitting:
        name, (pre_ms, post_ms) = fitting[0]
        disagreeing = [other for other, durations in fitting[1:]
                       if durations[0] != pre_ms]
        if disagreeing:
            print(f"  WARNING: {name} puts stimulus delivery {pre_ms} ms into "
                  f"the LFP epochs, but {', '.join(disagreeing)} fit the "
                  f"arrays with a different pre-stimulus time. Using {name}.")
        return {
            "pre_stim_s": pre_ms / 1000.0,
            "post_stim_s": post_ms / 1000.0,
            "total_s": n_samples / fs,
            "fs": fs,
        }

    tried = "; ".join(f"{name} = {value}" for name, value in candidates)
    raise ValueError(
        f"Cannot work out where stimulus delivery falls in the LFP epochs: "
        f"the arrays have {n_samples} samples at {fs} Hz, which matches none "
        f"of the known [pre, post] durations in ms ({tried}). If the LFPs were "
        f"not extracted at {fs} Hz this pipeline cannot be used on them as "
        f"is; otherwise set spike_array_durations in the .params file (or "
        f"ephys_data.lfp_params['trial_durations']) to the values used when "
        f"the LFPs were extracted.")


def make_stft_params(timing, post_stim_limit_s=None):
    """STFT settings for ephys_data.get_stft.

    By default the STFT covers the full LFP epoch. post_stim_limit_s stops it
    that many seconds after stimulus delivery instead (never beyond the end
    of the epoch), which is cheaper when later times are not needed.
    """
    end_s = timing["total_s"]
    if post_stim_limit_s is not None:
        end_s = min(end_s, timing["pre_stim_s"] + post_stim_limit_s)
    window = int(round(STFT_WINDOW_SECONDS * timing["fs"]))
    return {
        "Fs": timing["fs"],
        "signal_window": window,
        "window_overlap": window - 1,
        "max_freq": STFT_MAX_FREQ_HZ,
        "time_range_tuple": (0, end_s),
    }


def compute_stft(dat, timing, post_stim_limit_s=None):
    """Calculate the STFT power of dat.lfp_array_list, in memory only.

    Results are set on dat (dat.amplitude_array_list, dat.freq_vec,
    dat.time_vec) and nothing is written to the HDF5 file. The STFT cannot
    handle NaN/inf, so any channel containing a non-finite value is set to
    zero first. Validation never approves such a channel, so it is never used
    in the analysis; it only keeps its place on the channel axis.
    """
    zeroed_positions = set()
    for array in dat.lfp_array_list:
        nonfinite_channels = ~np.isfinite(array).all(axis=(0, 2))
        array[:, nonfinite_channels, :] = 0
        zeroed_positions.update(np.flatnonzero(nonfinite_channels).tolist())
    if zeroed_positions:
        print(f"  Zeroing LFP positions {sorted(zeroed_positions)} (non-finite "
              f"values) for the STFT; they are not used in the analysis")
    dat.stft_params = make_stft_params(timing, post_stim_limit_s)
    dat.get_stft(recalculate=True, write_out=False)


def get_taste_names_in_lfp_order(dat):
    """Return the taste name of every array in dat.lfp_array_list, in order.

    Each parsed LFP array is stored under its dig-in name
    (/Parsed_LFP/<dig_in_name>_LFPs) and the .info file maps dig-in names to
    taste names, so the taste of an array is looked up by dig-in name rather
    than by position. Any number of tastes is supported. The nodes are
    selected exactly as ephys_data.get_lfps selects them, so the two lists
    line up one to one (this is checked).
    """
    dat.get_info_dict()
    taste_params = dat.info_dict["taste_params"]
    if "dig_in_names" not in taste_params:
        raise ValueError(
            "The .info file has no taste_params['dig_in_names'], so parsed "
            "LFP arrays cannot be matched to tastes. Re-run blech_exp_info.")
    taste_by_dig_in = dict(zip(
        taste_params["dig_in_names"], taste_params["tastes"]))

    with tables.open_file(dat.hdf5_path, "r") as hf5:
        node_names = [node._v_name for node in hf5.list_nodes("/Parsed_LFP")
                      if "DIN" in node.__str__()]
    if len(node_names) != len(dat.lfp_array_list):
        raise ValueError(
            f"Found {len(node_names)} parsed LFP dig-ins but "
            f"{len(dat.lfp_array_list)} loaded LFP arrays")

    taste_names = []
    for node_name in node_names:
        dig_in_name = (node_name[:-len("_LFPs")]
                       if node_name.endswith("_LFPs") else node_name)
        if dig_in_name not in taste_by_dig_in:
            raise ValueError(
                f"Parsed LFP node {node_name!r} has no matching taste in the "
                f".info file (known dig-ins: {sorted(taste_by_dig_in)})")
        taste_names.append(taste_by_dig_in[dig_in_name])
    return taste_names


def baseline_window_mask(relative_time, baseline_seconds,
                         end_buffer_seconds=BASELINE_END_BUFFER_SECONDS):
    """Boolean mask of the pre-stimulus baseline window of a time vector.

    relative_time is in seconds with stimulus delivery at 0. The window runs
    from -baseline_seconds to -end_buffer_seconds.
    """
    mask = ((relative_time >= -baseline_seconds)
            & (relative_time <= -end_buffer_seconds))
    if not mask.any():
        raise ValueError(
            f"No STFT time points between -{baseline_seconds:g}s and "
            f"-{end_buffer_seconds:g}s before stimulus delivery; the "
            f"pre-stimulus part of the LFP epoch is too short for a baseline.")
    return mask


def post_stimulus_window_mask(relative_time):
    """Boolean mask of 0 to PLOT_POST_STIM_SECONDS after stimulus delivery."""
    return (relative_time >= 0) & (relative_time <= PLOT_POST_STIM_SECONDS)


def normalize_trial_power(amplitude_array, relative_time, baseline_seconds,
                          end_buffer_seconds=BASELINE_END_BUFFER_SECONDS):
    """Z-score STFT power to each trial's own pre-stimulus baseline.

    amplitude_array is (trials, channels, frequencies, time). Every
    trial/channel/frequency is normalized separately, using the mean and
    standard deviation of its own baseline window, before anything is
    averaged across trials, channels, or tastes.
    """
    baseline = amplitude_array[
        ..., baseline_window_mask(
            relative_time, baseline_seconds, end_buffer_seconds)]
    baseline_mean = np.mean(baseline, axis=-1, keepdims=True)
    # Floor the std at machine epsilon to avoid dividing by zero
    baseline_std = np.maximum(
        np.std(baseline, axis=-1, keepdims=True), np.finfo(float).eps)
    return (amplitude_array - baseline_mean) / baseline_std


def summarize_normalized_power(normalized, relative_time):
    """Reduce baseline-normalized power to the quantities that get plotted.

    normalized is (trials, channels, frequencies, time), from
    normalize_trial_power. Returns the median spectrogram over trials and
    channels (frequencies x time), and the mean power per frequency over
    trials, channels and time inside the baseline window ("pre_power") and
    the post-stimulus window ("post_power").
    """
    baseline_mask = baseline_window_mask(relative_time, PLOT_BASELINE_SECONDS)
    post_mask = post_stimulus_window_mask(relative_time)
    return {
        "spectrogram": np.median(normalized, axis=(0, 1)),
        "pre_power": np.mean(normalized[..., baseline_mask], axis=(0, 1, 3)),
        "post_power": np.mean(normalized[..., post_mask], axis=(0, 1, 3)),
    }


def load_cleaned_lfps(dat, data_dir, taste_names):
    """Swap dat.lfp_array_list for the cleaned, artifact-free trials.

    dat.get_lfps() must have been called, and remove_lfp_artifacts.py must
    have been run for this dataset. Tastes the user excluded there are left
    out. The cleaned arrays are checked against the original data and against
    the removal report, so stale files from a different run are caught.

    Sets dat.lfp_array_list (cleaned arrays), dat.active_taste_indices
    (position of each kept taste among all tastes) and dat.active_taste_names.
    """
    cleaned_dir = os.path.join(data_dir, CLEANED_LFP_FOLDER)
    report_path = os.path.join(cleaned_dir, "artifact_removal_report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"Missing {report_path}. Run remove_lfp_artifacts.py first.")
    with open(report_path, "r") as report_file:
        report = json.load(report_file)
    excluded = set(report.get("excluded_taste_indices", []))
    taste_reports = {
        taste["taste_index"]: taste for taste in report.get("tastes", [])}
    if len(taste_reports) != len(dat.lfp_array_list):
        raise ValueError(
            f"{report_path} describes {len(taste_reports)} tastes but the "
            f"dataset has {len(dat.lfp_array_list)}. Re-run "
            f"remove_lfp_artifacts.py.")

    cleaned_arrays = []
    active_indices = []
    active_names = []
    for taste_index, original in enumerate(dat.lfp_array_list):
        if taste_index in excluded:
            print(f"  Skipping excluded taste {taste_names[taste_index]} "
                  f"(index {taste_index}) before STFT")
            continue
        cleaned_path = os.path.join(
            cleaned_dir, f"taste_{taste_index:02d}_cleaned.npy")
        if not os.path.exists(cleaned_path):
            raise FileNotFoundError(
                f"Missing cleaned LFP file: {cleaned_path}. "
                "Run remove_lfp_artifacts.py first.")
        cleaned = np.load(cleaned_path)
        # (trials, channels, samples): trials may be removed, nothing else
        if cleaned.shape[1:] != original.shape[1:]:
            raise ValueError(
                f"Cleaned LFP shape mismatch for taste {taste_index}: "
                f"{cleaned.shape} versus {original.shape}")
        if list(cleaned.shape) != taste_reports[taste_index]["output_shape"]:
            raise ValueError(
                f"Cleaned LFP {cleaned_path} does not match its artifact "
                f"report")
        cleaned_arrays.append(cleaned)
        active_indices.append(taste_index)
        active_names.append(taste_names[taste_index])
        print(f"  Loaded cleaned taste {taste_names[taste_index]} "
              f"(index {taste_index}): "
              f"{cleaned.shape[0]}/{original.shape[0]} trials")

    dat.lfp_array_list = cleaned_arrays
    dat.active_taste_indices = np.asarray(active_indices, dtype=int)
    dat.active_taste_names = active_names
