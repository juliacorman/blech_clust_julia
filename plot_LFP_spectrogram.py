#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 14:22:19 2026

@author: cmazzio
"""
# IMPORTS
## Run this script from blech_spyder environment and within blech_clust dir ##


import matplotlib
matplotlib.use('Agg')

import blech_clust.utils.ephys_data.ephys_data as ephys_data
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import zscore
import os
import pickle
import json
from validate_LFP_data import validate_dataset, write_validation_text, json_default

#%% Direct to Data

file_path = "/home/julia-corman/Desktop/Personal_Data/data_dirs_LFP.txt"
CLEANED_LFP_FOLDER = "cleaned_LFP_artifacts"

with open(file_path, "r") as f:
    data_dirs = [line.strip() for line in f.readlines()]

print(f"Found {len(data_dirs)} directories:")
print(data_dirs)


#%% Function Definition

validated_region_maps = {}


def load_cleaned_lfps(dat, data_dir):
    """Load cleaned trial copies while preserving original channel positions."""
    cleaned_dir = os.path.join(data_dir, CLEANED_LFP_FOLDER)
    cleaned_arrays = []
    excluded_taste_indices = set()
    report_path = os.path.join(cleaned_dir, "artifact_removal_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as report_file:
            excluded_taste_indices = set(
                json.load(report_file).get("excluded_taste_indices", []))
    for taste_index, original in enumerate(dat.lfp_array_list):
        cleaned_path = os.path.join(
            cleaned_dir, f"taste_{taste_index:02d}_cleaned.npy")
        if not os.path.exists(cleaned_path):
            raise FileNotFoundError(
                f"Missing cleaned LFP file: {cleaned_path}. "
                "Run remove_lfp_artifacts.py first.")
        cleaned = np.load(cleaned_path)
        if cleaned.shape[1:] != original.shape[1:]:
            raise ValueError(f"Cleaned LFP shape mismatch for taste {taste_index}: "
                             f"{cleaned.shape} versus {original.shape}")
        cleaned_arrays.append(cleaned)
        print(f"  Loaded cleaned taste {taste_index}: "
              f"{cleaned.shape[0]}/{original.shape[0]} trials")
    dat.lfp_array_list = cleaned_arrays
    return excluded_taste_indices


def baseline_normalize_spectrogram(spectrogram, time_vec):
    """Z-score each frequency using only its pre-stimulus baseline."""
    baseline_mask = (time_vec >= -2) & (time_vec <= 0)
    baseline = spectrogram[:, baseline_mask]
    baseline_mean = np.mean(baseline, axis=1, keepdims=True)
    baseline_std = np.std(baseline, axis=1, keepdims=True)
    baseline_std = np.maximum(baseline_std, np.finfo(float).eps)
    return (spectrogram - baseline_mean) / baseline_std


def normalize_trial_power(amplitude_array, time_vec):
    """Normalize STFT power per trial/channel/frequency to its baseline in dB."""
    baseline_mask = (time_vec >= -2) & (time_vec <= 0)
    baseline_power = np.mean(
        amplitude_array[:, :, :, baseline_mask], axis=-1, keepdims=True)
    baseline_power = np.maximum(baseline_power, np.finfo(float).eps)
    return 10 * np.log10(np.maximum(amplitude_array, np.finfo(float).eps)
                         / baseline_power)


def relative_time_vector(dat):
    """Convert the STFT 0-to-5-second vector to -2-to-5-second time."""
    return dat.time_vec - 2.0


def taste_broadband_qc(dat, good_trial_masks, taste_list):
    """Exclude tastes with extreme broadband RMS after trial QC."""
    taste_rms = []
    for lfp_data, good_trials in zip(dat.lfp_array_list, good_trial_masks):
        retained = lfp_data[good_trials]
        taste_rms.append(float(np.sqrt(np.mean(retained ** 2))))

    active_tastes = np.ones(len(taste_rms), dtype=bool)
    for taste_index, rms in enumerate(taste_rms):
        comparison = np.asarray(taste_rms)[np.arange(len(taste_rms)) != taste_index]
        if len(comparison) < 2:
            continue
        center = np.median(comparison)
        scale = 1.4826 * np.median(np.abs(comparison - center))
        if scale > 0 and rms > center + 3 * scale:
            active_tastes[taste_index] = False
            print(f"  WARNING: excluding taste {taste_list[taste_index]} "
                  f"for broadband artifact (RMS {rms:.2f}; "
                  f"other-taste limit {center + 3 * scale:.2f})")

    dat.analysis_taste_indices = np.where(active_tastes)[0]
    return active_tastes, taste_rms


def apply_lfp_trial_artifact_filter(dat, mad_threshold=3):
    """Remove trials flagged by blech_clust's MAD-based LFP artifact check."""
    try:
        from blech_clust.utils.ephys_data import lfp_processing
    except ModuleNotFoundError:
        from utils.ephys_data import lfp_processing

    good_trial_masks = []
    filtered_amplitudes = []
    for taste, lfp_data in enumerate(dat.lfp_array_list):
        channel_trial_data = lfp_data.swapaxes(0, 1)
        good_trials = lfp_processing.return_good_lfp_trial_inds(
            channel_trial_data, MAD_threshold=mad_threshold)
        if not np.any(good_trials):
            print(f"  WARNING: no good trials for taste {taste}; keeping all trials")
            good_trials = np.ones(lfp_data.shape[0], dtype=bool)
        good_trial_masks.append(good_trials)
        filtered_amplitudes.append(dat.amplitude_array_list[taste][good_trials])
        print(f"  Taste {taste}: retained {np.sum(good_trials)}/"
              f"{len(good_trials)} trials after artifact rejection")

    dat.amplitude_array_list = filtered_amplitudes
    return good_trial_masks


def get_lfp_region_map(dat, data_dir=None):
    """Return parsed-LFP positions grouped by region and validation details."""
    dat.get_info_dict()
    dat.get_lfp_channels()
    dat.get_region_electrodes()

    region_map = {region: [] for region in dat.region_electrode_dict}
    assignments = []
    for lfp_position, electrode in enumerate(dat.parsed_lfp_channels):
        matches = []
        for region, car_groups in dat.region_electrode_dict.items():
            if any(electrode in car_group for car_group in car_groups):
                matches.append(region)

        if len(matches) == 1:
            region_map[matches[0]].append(lfp_position)
        assignments.append({
            "lfp_position": int(lfp_position),
            "electrode": int(electrode),
            "regions": matches,
            "status": "assigned" if len(matches) == 1 else "unassigned_or_ambiguous"
        })

    region_map = {region: np.asarray(indices, dtype=int)
                  for region, indices in region_map.items()}
    if data_dir in validated_region_maps:
        approved = validated_region_maps[data_dir]
        region_map = {
            region: np.asarray(approved.get(region, []), dtype=int)
            for region in region_map
        }
    return region_map, assignments


def validate_lfp_regions(data_dirs_list, output_path=None):
    """Print and optionally save the region assignment for every dataset."""
    validation = []
    for data_dir in data_dirs_list:
        basename = os.path.basename(data_dir)
        result = validate_dataset(data_dir)
        validation.append(result)
        write_validation_text(data_dir, result)
        validated_region_maps[data_dir] = {
            region: np.asarray(indices, dtype=int)
            for region, indices in result["good_lfp_positions_by_region"].items()
        }

        print(f"\nRegion validation: {basename}")
        for summary in result["region_summaries"]:
            print(f"  {summary['region']}: mean={summary['mean_voltage']:.3f}, "
                  f"median={summary['median_voltage']:.3f}, "
                  f"std={summary['voltage_standard_deviation']:.3f} "
                  f"({summary['channel_count']} good channels)")
        bad_rows = [row for row in result["channels"]
                    if not row["use_for_analysis"]]
        if bad_rows:
            print("  BAD CHANNELS:")
            for row in bad_rows:
                print(f"    electrode {row['electrode']} "
                      f"({row['region'] or 'unknown'}): "
                      f"{row['quality_status']}")
        else:
            print("  No bad channels detected.")

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(validation, f, indent=2, default=json_default)
        print(f"Region validation saved to: {output_path}")

    return validation

def process_session(data_dir):
    basename = os.path.basename(data_dir)
    print(f"\nProcessing session: {basename}")

    # Step 1: Load one epoched dataset. The source epochs already span
    # approximately -2 to +5 seconds around the behavioral event.
    dat = ephys_data.ephys_data(data_dir)

    # Step 2: Keep the existing STFT settings. STFT calculation uses the
    # full epoch; plotting later restricts the displayed time range.
    dat.stft_params = {
        'Fs': 1000,
        'signal_window': 500,
        'window_overlap': 499,
        'max_freq': 40,
        'time_range_tuple': (0, 5)
    }

    # Step 3: Load cleaned trial copies produced by remove_lfp_artifacts.py.
    # This leaves the original dataset unchanged.
    dat.get_lfps()
    excluded_tastes = load_cleaned_lfps(dat, data_dir)

    # Step 4: Calculate the existing STFT from the retained trials in memory.
    # write_out=False prevents cleaned results from being written to the source.
    dat.get_stft(recalculate=True, write_out=False)
    dat.get_info_dict()
    info_dict = dat.info_dict

    taste_list = info_dict['taste_params']['tastes']
    active_tastes = np.ones(len(taste_list), dtype=bool)
    active_tastes[list(excluded_tastes)] = False
    amp_array_shape = [x.shape for x in dat.amplitude_array_list]

    print("Frequencies:", dat.freq_vec)

    # Set save directory
    save_dir = os.path.join(data_dir, 'LFP_analyses')
    os.makedirs(save_dir, exist_ok=True)

    # Save variables
    save_path = os.path.join(save_dir, f"{basename}_STFT_data.pkl")
    data_to_save = {
        "amplitude_array_list": dat.amplitude_array_list,
        "freq_vec": dat.freq_vec,
        "time_vec": dat.time_vec,
        "amp_array_shape": amp_array_shape,
        "info_dict": info_dict,
        "taste_list": taste_list
    }

    with open(save_path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"All variables saved to {save_path}")

    # Step 5: Select validated channels by implanted region.
    region_map, _ = get_lfp_region_map(dat, data_dir)
    active_taste_indices = np.where(active_tastes)[0]
    num_tastes = len(active_taste_indices)
    relative_time = relative_time_vector(dat)
    time_vec_ms = relative_time * 1000
    y_ticks = np.arange(dat.freq_vec.min(), dat.freq_vec.max() + 1, 5)
    x_min, x_max = -1000, 1750
    x_ticks = np.arange(x_min, x_max + 1, 250)
    # Step 6: Restrict figures to -1 second through +1.75 seconds.
    time_mask = (time_vec_ms >= x_min) & (time_vec_ms <= x_max)

    for region, channel_indices in region_map.items():
        if len(channel_indices) == 0:
            continue
        # Step 7: Normalize every trial independently at every frequency using
        # its -2 to 0 second baseline before any trial averaging.
        region_data = [normalize_trial_power(
                   dat.amplitude_array_list[t][:, channel_indices, :, :],
                           relative_time)
                       for t in active_taste_indices]
        # Step 8: Average the normalized trials and channels for each taste.
        spectrograms = [np.mean(stft_dat, axis=(0, 1))
                        for stft_dat in region_data]
        for normalized, suffix, colorbar_label in [
            (False, "", "Baseline-normalized power (dB)"),
                (True, "_baseline_zscore", "Baseline z-scored power")]:
            plotted = [baseline_normalize_spectrogram(spec, relative_time)
                       if normalized else spec for spec in spectrograms]
            all_data = np.concatenate([spec.flatten() for spec in plotted])
            vmin, vmax = np.min(all_data), np.max(all_data)
            fig, axes = plt.subplots(
                num_tastes, 1, figsize=(12, 8 * num_tastes), sharex=True)
            if num_tastes == 1:
                axes = [axes]

            for taste, ax in enumerate(axes):
                pcm = ax.pcolormesh(
                    time_vec_ms[time_mask], dat.freq_vec,
                    plotted[taste][:, time_mask], shading='auto', cmap='jet',
                    vmin=vmin, vmax=vmax)
                ax.axvline(0, color='red', linewidth=2, linestyle='--')
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)
                ax.tick_params(axis='both', labelsize=16)
                ax.set_ylabel("")
                ax.set_title(
                    f"{taste_list[active_taste_indices[taste]]} | {basename} | "
                    f"{region} channels: {len(channel_indices)}", fontsize=22)

            axes[-1].set_xlabel("Time (ms)", fontsize=22)
            fig.text(0.05, 0.5, "Frequency (Hz)", va='center',
                     rotation='vertical', fontsize=22)
            fig.subplots_adjust(right=0.88, hspace=0.5)
            cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(pcm, cax=cbar_ax, label=colorbar_label)
            cbar.ax.tick_params(labelsize=16)
            cbar.ax.yaxis.label.set_size(16)

            fig_name_base = os.path.join(
                save_dir, f'{basename}_{region}_STFT_spectrogram{suffix}')
            fig.savefig(f'{fig_name_base}.svg', dpi=300, bbox_inches='tight')
            fig.savefig(f'{fig_name_base}.png', dpi=300, bbox_inches='tight')
            print(f"{region} spectrogram figures saved to: {fig_name_base}")
            plt.close(fig)


#%% Function for power spectrum analysis

def plot_power_spectrum(data_dir):
    basename = os.path.basename(data_dir)
    print(f"\nProcessing power spectrum for session: {basename}")

    # Initialize dat object
    dat = ephys_data.ephys_data(data_dir)

    # Set STFT parameters
    dat.stft_params = {
        'Fs': 1000,
        'signal_window': 500,
        'window_overlap': 499,
        'max_freq': 40,
        'time_range_tuple': (0, 5)
    }

    # Run required pipeline steps
    dat.get_lfps()
    excluded_tastes = load_cleaned_lfps(dat, data_dir)
    dat.get_stft(recalculate=True, write_out=False)
    dat.get_info_dict()
    info_dict = dat.info_dict
    taste_list = info_dict['taste_params']['tastes']
    active_tastes = np.ones(len(taste_list), dtype=bool)
    active_tastes[list(excluded_tastes)] = False

    # Set save directory
    save_dir = os.path.join(data_dir, 'LFP_analyses')
    os.makedirs(save_dir, exist_ok=True)

    # Frequency and time masks
    freq_mask = (dat.freq_vec >= 0) & (dat.freq_vec <= 35)
    relative_time = relative_time_vector(dat)
    time_pre_mask = (relative_time >= -2) & (relative_time <= 0)
    time_post_mask = (relative_time >= 0) & (relative_time <= 1.75)

    region_map, _ = get_lfp_region_map(dat, data_dir)
    num_tastes = len(dat.amplitude_array_list)

    for region, channel_indices in region_map.items():
        if len(channel_indices) == 0:
            continue
        pre_power_list = []
        post_power_list = []
        for taste in np.where(active_tastes)[0]:
            amp_array = normalize_trial_power(
                dat.amplitude_array_list[taste][:, channel_indices, :, :],
                relative_time)
            for trial in range(amp_array.shape[0]):
                power = amp_array[trial]
                pre_avg = np.mean(power[:, :, time_pre_mask], axis=(0, 2))
                post_avg = np.mean(power[:, :, time_post_mask], axis=(0, 2))
                pre_power_list.append(pre_avg[freq_mask])
                post_power_list.append(post_avg[freq_mask])

        pre_power = np.mean(np.array(pre_power_list), axis=0)
        post_power = np.mean(np.array(post_power_list), axis=0)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(dat.freq_vec[freq_mask], pre_power,
                label='Pre-stimulus (1s)', linewidth=2)
        ax.plot(dat.freq_vec[freq_mask], post_power,
                label='Post-stimulus (1s)', linewidth=2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Baseline-normalized power (dB)')
        ax.set_title(f'{basename} - {region} Average Power Spectrum 0-35 Hz')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.legend()
        ax.grid(True, alpha=0.3)

        power_fig_base = os.path.join(
            save_dir, f'{basename}_{region}_power_spectrum_0-35Hz')
        fig.savefig(f'{power_fig_base}.svg', dpi=300, bbox_inches='tight')
        fig.savefig(f'{power_fig_base}.png', dpi=300, bbox_inches='tight')
        print(f"{region} power spectrum figures saved to: {power_fig_base}")
        plt.close(fig)


#%% Function for grand average power spectrum across multiple sessions

def plot_avg_power_spectrum(data_dirs_list, output_dir):
    print(f"\nComputing region-specific grand averages across {len(data_dirs_list)} sessions")
    os.makedirs(output_dir, exist_ok=True)
    region_data = {}
    freq_vec_ref = None
    time_vec_ref = None

    for data_dir in data_dirs_list:
        basename = os.path.basename(data_dir)
        print(f"Processing: {basename}")
        try:
            dat = ephys_data.ephys_data(data_dir)
            dat.stft_params = {
                'Fs': 1000,
                'signal_window': 500,
                'window_overlap': 499,
                'max_freq': 40,
                'time_range_tuple': (0, 5)
            }
            dat.get_lfps()
            excluded_tastes = load_cleaned_lfps(dat, data_dir)
            dat.get_stft(recalculate=True, write_out=False)
            dat.get_info_dict()
            taste_list = dat.info_dict['taste_params']['tastes']
            active_tastes = np.ones(len(taste_list), dtype=bool)
            active_tastes[list(excluded_tastes)] = False
            region_map, assignments = get_lfp_region_map(dat, data_dir)

            if freq_vec_ref is None:
                freq_vec_ref = dat.freq_vec
                time_vec_ref = dat.time_vec
            freq_mask = (dat.freq_vec >= 0) & (dat.freq_vec <= 35)
            relative_time = relative_time_vector(dat)
            time_pre_mask = (relative_time >= -2) & (relative_time <= 0)
            time_post_mask = (relative_time >= 0) & (relative_time <= 1.75)

            for region, channel_indices in region_map.items():
                if len(channel_indices) == 0:
                    print(f"  Skipping {region}: no parsed LFP channels")
                    continue
                pre_power_list = []
                post_power_list = []
                spectrograms = []
                for taste in np.where(active_tastes)[0]:
                    amp_array = dat.amplitude_array_list[taste]
                    region_amp = amp_array[:, channel_indices, :, :]
                    for trial in range(region_amp.shape[0]):
                        power = region_amp[trial]
                        pre_power_list.append(
                            np.mean(power[:, :, time_pre_mask], axis=(0, 2))[
                                freq_mask])
                        post_power_list.append(
                            np.mean(power[:, :, time_post_mask], axis=(0, 2))[
                                freq_mask])
                    med_stft = np.median(
                        normalize_trial_power(region_amp, relative_time), axis=(0, 1))
                    spectrograms.append(med_stft)

                region_data.setdefault(region, {
                    "pre_power": [], "post_power": [], "spectrograms": [],
                    "sessions": []
                })
                region_data[region]["pre_power"].append(np.mean(pre_power_list, axis=0))
                region_data[region]["post_power"].append(np.mean(post_power_list, axis=0))
                session_spectrogram = np.mean(spectrograms, axis=0)
                session_mean = np.mean(session_spectrogram, axis=1, keepdims=True)
                session_std = np.std(session_spectrogram, axis=1, keepdims=True)
                session_std = np.maximum(session_std, np.finfo(float).eps)
                region_data[region]["spectrograms"].append(
                    (session_spectrogram - session_mean) / session_std)
                region_data[region]["sessions"].append(basename)
        except Exception as e:
            print(f"Error processing {basename}: {e}")

    freq_subset = freq_vec_ref[(freq_vec_ref >= 0) & (freq_vec_ref <= 35)]
    time_vec_ms = (time_vec_ref - 2.0) * 1000
    x_min, x_max = -1000, 1750
    time_mask = (time_vec_ms >= x_min) & (time_vec_ms <= x_max)

    for region, values in region_data.items():
        safe_region = region.replace(' ', '_')
        grand_pre_power = np.mean(values["pre_power"], axis=0)
        grand_post_power = np.mean(values["post_power"], axis=0)
        grand_spectrogram = np.mean(values["spectrograms"], axis=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(freq_subset, grand_pre_power, label='Pre-stimulus (1s)', linewidth=2)
        ax.plot(freq_subset, grand_post_power, label='Post-stimulus (1s)', linewidth=2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f'{region} Average Power Spectrum 0-35 Hz '
                     f'({len(values["sessions"])} sessions)')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.legend()
        ax.grid(True, alpha=0.3)
        power_base = os.path.join(output_dir,
                                  f'grand_average_{safe_region}_power_spectrum_0-35Hz')
        fig.savefig(f'{power_base}.svg', dpi=300, bbox_inches='tight')
        fig.savefig(f'{power_base}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        for normalized, suffix, colorbar_label in [
                (False, "", "Power"),
                (True, "_baseline_zscore", "Baseline z-scored power")]:
            plotted_spectrogram = (
                baseline_normalize_spectrogram(grand_spectrogram, time_vec_ref)
                if normalized else grand_spectrogram)
            vmin, vmax = np.min(plotted_spectrogram), np.max(plotted_spectrogram)
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            pcm = ax2.pcolormesh(
                time_vec_ms[time_mask], freq_vec_ref,
                plotted_spectrogram[:, time_mask], shading='auto', cmap='jet',
                vmin=vmin, vmax=vmax)
            ax2.axvline(0, color='red', linewidth=2, linestyle='--')
            ax2.set_xticks(np.arange(x_min, x_max + 1, 250))
            ax2.set_yticks(np.arange(freq_vec_ref.min(), freq_vec_ref.max() + 1, 5))
            ax2.tick_params(axis='both', labelsize=16)
            ax2.set_xlabel("Time (ms)", fontsize=22)
            ax2.set_ylabel("Frequency (Hz)", fontsize=22)
            ax2.set_title(f'{region} Average Spectrogram '
                          f'({len(values["sessions"])} sessions, active tastes)',
                          fontsize=22)
            cbar = fig2.colorbar(pcm, ax=ax2, label=colorbar_label)
            cbar.ax.tick_params(labelsize=16)
            spec_base = os.path.join(
                output_dir, f'grand_average_{safe_region}_spectrogram{suffix}')
            fig2.savefig(f'{spec_base}.svg', dpi=300, bbox_inches='tight')
            fig2.savefig(f'{spec_base}.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
        print(f"Saved {region} averages for {len(values['sessions'])} sessions")


def main():
    # Step 12: Validate all datasets before any session plots are created.
    validation_dir = '/home/julia-corman/Desktop/Personal_Data/grand_average_results'
    os.makedirs(validation_dir, exist_ok=True)
    validate_lfp_regions(
        data_dirs,
        os.path.join(validation_dir, 'lfp_region_validation.json')
    )

    # Step 13: Create individual, region-specific session plots.
    for dir_path in data_dirs:
        process_session(dir_path)
        plot_power_spectrum(dir_path)

    # Grand averages are generated separately by grand_average_LFP.py.


if __name__ == "__main__":
    main()

#%% Save STFT data and plot spectrogram for all data dirs

#%%


# # Set data directory
# data_dir = '/media/cmazzio/large_data/CM77/CM77_CTATest2_h2o_nacl_lowqhcl_highqhcl_250627_103549'
# basename = "CM77_CTATest2"
# # Intialize dat variable with data directory
# dat = ephys_data.ephys_data(data_dir)

# # Calculate Short-Time Fourier Transform (STFT)
# # Default params in script
# dat.get_stft()

# # Make sure STFT ran properly

# # Gives you frequency bands detected in data
# print(dat.freq_vec)

# # Gives you time points (based on params)
# print(dat.time_vec)

# # Gives you shape of each amplitude array for each taste
# # Should be shape : trials x num channels (electrodes with signal) x number of frequencies x timepoints
# amp_array_shape = [x.shape for x in dat.amplitude_array_list]
# print(amp_array_shape)

# # Save info file dictionary 
# dat.get_info_dict()
# info_dict = dat.info_dict

# # Taste names saved under 'tastes' key of dictionary
# this_session_taste_list = info_dict['taste_params']['tastes']
# print(this_session_taste_list)

# # Set save dir for all files and plots
# save_dir = os.path.join(data_dir, 'LFP_analyses')
# # Create the directory if it doesn't exist
# os.makedirs(save_dir, exist_ok=True)

# # File to save all variables
# save_path = os.path.join(save_dir, f"{basename}_STFT_data.pkl")

# # Variables to save
# data_to_save = {
#     "amplitude_array_list": dat.amplitude_array_list,
#     "freq_vec": dat.freq_vec,
#     "time_vec": dat.time_vec,
#     "amp_array_shape": amp_array_shape,
#     "info_dict": info_dict,
#     "taste_list": this_session_taste_list
# }

# # Save with pickle
# with open(save_path, "wb") as f:
#     pickle.dump(data_to_save, f)

# print(f"All variables saved to {save_path}")

# #%% Plot Spectrogram for all tastes in session

# num_tastes = len(dat.amplitude_array_list)

# fig, axes = plt.subplots(num_tastes, 1, figsize=(12, 8*num_tastes), sharex=True)

# # Ensure axes is always iterable
# if num_tastes == 1:
#     axes = [axes]

# time_vec_ms = (dat.time_vec * 1000) - 2000

# # Frequency ticks
# y_min, y_max = dat.freq_vec.min(), dat.freq_vec.max()
# y_ticks = np.arange(y_min, y_max+1, 5)  # adjust interval as needed

# # X-axis ticks
# x_min, x_max = -500, 2000
# x_ticks = np.arange(x_min, x_max+1, 250)

# for taste, ax in enumerate(axes):

#     stft_dat = dat.amplitude_array_list[taste]
#     med_stft = np.median(stft_dat, axis=(0,1))
#     num_channels = stft_dat.shape[1]

#     # Keep only time >= -500 ms
#     time_mask = (time_vec_ms >= x_min) & (time_vec_ms <= x_max)

#     pcm = ax.pcolormesh(
#         time_vec_ms[time_mask],
#         dat.freq_vec,
#         zscore(med_stft[:, time_mask], axis=-1),
#         shading='auto',
#     )

#     ax.axvline(0, color='red', linewidth=2, linestyle='--')

#     # Tick labels and size
#     ax.tick_params(axis='both', labelsize=16)
#     ax.set_xticks(x_ticks)
#     ax.set_yticks(y_ticks)

#     # Remove individual y-axis label
#     ax.set_ylabel("")

#     # Title
#     taste_name = this_session_taste_list[taste]
#     ax.set_title(f"{taste_name} | {basename} | Channels: {num_channels}", fontsize=22)

# # Shared labels
# axes[-1].set_xlabel("Time (ms)", fontsize=22)
# fig.text(0.05, 0.5, "Frequency (Hz)", va='center', rotation='vertical', fontsize=22)

# # Leave space on right for colorbar
# fig.subplots_adjust(right=0.88, hspace=0.5)

# # Add colorbar
# cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
# cbar = fig.colorbar(pcm, cax=cbar_ax, label="Z-scored power")
# cbar.ax.tick_params(labelsize = 16)
# cbar.ax.yaxis.label.set_size(16)

# fig_name_svg = f'{basename}_STFT_spectrogram.svg'
# fig_name_png = f'{basename}_STFT_spectrogram.png'
# save_path_svg = os.path.join(save_dir, fig_name_svg)
# save_path_png = os.path.join(save_dir, fig_name_png)
# plt.savefig(save_path_svg, dp=300, bbox_inches = 'tight')
# plt.savefig(save_path_png, dp=300, bbox_inches = 'tight')
# print(f"Figures saved to: {save_path}")
                         
# plt.show()
