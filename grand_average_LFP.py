#!/usr/bin/env python3
"""Create region-specific grand averages from cleaned, normalized LFP trials."""

import json
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

import blech_clust.utils.ephys_data.ephys_data as ephys_data
from validate_LFP_data import validate_dataset

DATA_LIST = "/home/julia-corman/Desktop/Personal_Data/data_dirs_LFP.txt"
CLEANED_FOLDER = "cleaned_LFP_artifacts"
OUTPUT_DIR = "/home/julia-corman/Desktop/Personal_Data/grand_average_results"


def normalize_trial_power(power, time_vec):
    # Step 1: Normalize each trial, channel, and frequency to its
    # -2 to 0 second pre-stimulus baseline.
    baseline_mask = (time_vec >= -2) & (time_vec <= 0)
    baseline = np.mean(power[:, :, :, baseline_mask], axis=-1, keepdims=True)
    baseline = np.maximum(baseline, np.finfo(float).eps)
    return 10 * np.log10(np.maximum(power, np.finfo(float).eps) / baseline)


def load_cleaned(dat, data_dir):
    # Step 2: Load the separate cleaned copy and preserve channel positions.
    cleaned_dir = os.path.join(data_dir, CLEANED_FOLDER)
    with open(os.path.join(cleaned_dir, "artifact_removal_report.json")) as f:
        excluded = set(json.load(f).get("excluded_taste_indices", []))
    arrays = []
    for taste, original in enumerate(dat.lfp_array_list):
        path = os.path.join(cleaned_dir, f"taste_{taste:02d}_cleaned.npy")
        cleaned = np.load(path)
        if cleaned.shape[1:] != original.shape[1:]:
            raise ValueError(f"Channel shape mismatch in {path}")
        arrays.append(cleaned)
    dat.lfp_array_list = arrays
    return excluded


def region_map(dat, data_dir):
    # Step 3: Keep only channels approved by validation, grouped by region.
    validation = validate_dataset(data_dir)
    return {
        region: np.asarray(indices, dtype=int)
        for region, indices in validation["good_lfp_positions_by_region"].items()
    }


def main():
    # Step 4: Collect one dataset path per line from the session list.
    with open(DATA_LIST) as f:
        data_dirs = [line.strip() for line in f if line.strip()]
    region_data = {}
    freq_ref = None
    time_ref = None

    for data_dir in data_dirs:
        dataset = os.path.basename(os.path.normpath(data_dir))
        dat = ephys_data.ephys_data(data_dir)
        # Step 5: Use the existing full-epoch STFT settings.
        dat.stft_params = {
            "Fs": 1000, "signal_window": 500, "window_overlap": 499,
            "max_freq": 40, "time_range_tuple": (0, 5)
        }
        # Step 6: Load cleaned trials and calculate STFTs in memory.
        dat.get_lfps()
        excluded = load_cleaned(dat, data_dir)
        dat.get_stft(recalculate=True, write_out=False)
        regions = region_map(dat, data_dir)
        if freq_ref is None:
            freq_ref, time_ref = dat.freq_vec, dat.time_vec

        for region, channels in regions.items():
            if len(channels) == 0:
                continue
            session_spectrograms = []
            session_pre = []
            session_post = []
            for taste, amp in enumerate(dat.amplitude_array_list):
                if taste in excluded:
                    continue
                # Step 7: Normalize each trial before averaging this taste.
                relative_time = dat.time_vec - 2.0
                normalized = normalize_trial_power(
                    amp[:, channels, :, :], relative_time)
                session_spectrograms.append(np.mean(normalized, axis=(0, 1)))
                pre = np.mean(normalized[:, :, :, (relative_time >= -2) &
                                           (relative_time <= 0)], axis=(0, 1, 3))
                post = np.mean(normalized[:, :, :, (relative_time >= 0) &
                                            (relative_time <= 1.75)], axis=(0, 1, 3))
                session_pre.append(pre)
                session_post.append(post)
            if not session_spectrograms:
                continue
            entry = region_data.setdefault(region, {"spectrograms": [], "pre": [], "post": [], "sessions": []})
            # Step 8: Average active tastes to create one session-level result.
            session_spectrogram = np.mean(session_spectrograms, axis=0)
            # Step 9: Z-score each frequency across time within this session
            # before combining sessions or animals.
            session_mean = np.mean(session_spectrogram, axis=1, keepdims=True)
            session_std = np.maximum(
                np.std(session_spectrogram, axis=1, keepdims=True),
                np.finfo(float).eps)
            entry["spectrograms"].append(
                (session_spectrogram - session_mean) / session_std)
            entry["pre"].append(np.mean(session_pre, axis=0))
            entry["post"].append(np.mean(session_post, axis=0))
            entry["sessions"].append(dataset)

    # Step 10: Grand-average the already normalized and session-z-scored data.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    time_ms = (time_ref - 2.0) * 1000
    mask = (time_ms >= -1000) & (time_ms <= 1750)
    freq_mask = (freq_ref >= 0) & (freq_ref <= 35)
    # Step 11: Save region-specific grand-average power and spectrogram plots.
    for region, values in region_data.items():
        safe = region.replace(" ", "_")
        spectrogram = np.mean(values["spectrograms"], axis=0)
        pre = np.mean(values["pre"], axis=0)
        post = np.mean(values["post"], axis=0)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(freq_ref[freq_mask], pre[freq_mask], label="Pre-stimulus")
        ax.plot(freq_ref[freq_mask], post[freq_mask], label="Post-stimulus")
        ax.set(xlabel="Frequency (Hz)", ylabel="Baseline-normalized power (dB)")
        ax.set_title(f"{region} Grand Average Power Spectrum")
        ax.legend(); ax.grid(alpha=0.3)
        base = os.path.join(OUTPUT_DIR, f"grand_average_{safe}_power_spectrum")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".svg", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        pcm = ax.pcolormesh(time_ms[mask], freq_ref, spectrogram[:, mask], shading="auto", cmap="jet")
        fig.colorbar(pcm, ax=ax, label="Baseline-normalized power (dB)")
        ax.axvline(0, color="red", linestyle="--")
        ax.set(xlabel="Time (ms)", ylabel="Frequency (Hz)")
        ax.set_title(f"{region} Grand Average Spectrogram")
        base = os.path.join(OUTPUT_DIR, f"grand_average_{safe}_spectrogram")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".svg", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {region} averages for {len(values['sessions'])} sessions")


if __name__ == "__main__":
    main()
