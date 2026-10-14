"""Tests for the helpers shared by the LFP validation/plotting scripts."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import tables

import lfp_common


def make_dat(n_samples, lfp_params=None, sorting_durations=None,
             n_tastes=1):
    """Minimal stand-in for an ephys_data object with loaded LFPs."""
    def get_sorting_params_dict():
        if sorting_durations is None:
            raise IndexError("no .params file")
        dat.sorting_params_dict = {
            "spike_array_durations": sorting_durations}

    dat = SimpleNamespace(
        lfp_array_list=[np.zeros((3, 2, n_samples)) for _ in range(n_tastes)],
        lfp_params=lfp_params or {
            "trial_durations": None, "fin_sampling_rate": None},
        default_lfp_params={
            "trial_durations": [2000, 5000], "fin_sampling_rate": 1000},
        get_sorting_params_dict=get_sorting_params_dict,
    )
    return dat


class TestEpochTiming:
    def test_default_durations(self):
        timing = lfp_common.get_lfp_epoch_timing(make_dat(7000))
        assert timing == {"pre_stim_s": 2.0, "post_stim_s": 5.0,
                          "total_s": 7.0, "fs": 1000}

    def test_params_file_durations(self):
        dat = make_dat(4000, sorting_durations=[1000, 3000])
        timing = lfp_common.get_lfp_epoch_timing(dat)
        assert timing["pre_stim_s"] == 1.0
        assert timing["total_s"] == 4.0

    def test_explicit_lfp_params_used_when_params_file_does_not_fit(self):
        dat = make_dat(
            3000, lfp_params={"trial_durations": [500, 2500],
                              "fin_sampling_rate": 1000})
        assert lfp_common.get_lfp_epoch_timing(dat)["pre_stim_s"] == 0.5

    def test_params_file_takes_priority_and_conflict_warns(self, capsys):
        dat = make_dat(
            7000, lfp_params={"trial_durations": [3000, 4000],
                              "fin_sampling_rate": 1000},
            sorting_durations=[2000, 5000])
        assert lfp_common.get_lfp_epoch_timing(dat)["pre_stim_s"] == 2.0
        assert "WARNING" in capsys.readouterr().out

    def test_other_extraction_rate_warns(self, capsys):
        dat = make_dat(
            7000, lfp_params={"trial_durations": None,
                              "fin_sampling_rate": 500})
        assert lfp_common.get_lfp_epoch_timing(dat)["fs"] == 1000
        assert "WARNING" in capsys.readouterr().out

    def test_default_rate_does_not_warn(self, capsys):
        lfp_common.get_lfp_epoch_timing(make_dat(7000))
        assert "WARNING" not in capsys.readouterr().out

    def test_unmatched_length_raises(self):
        with pytest.raises(ValueError, match="stimulus delivery"):
            lfp_common.get_lfp_epoch_timing(make_dat(6000))


class TestStftParams:
    def test_full_epoch(self):
        timing = {"pre_stim_s": 2.0, "total_s": 7.0, "fs": 1000}
        params = lfp_common.make_stft_params(timing)
        assert params == {
            "Fs": 1000, "signal_window": 500, "window_overlap": 499,
            "max_freq": 40, "time_range_tuple": (0, 7.0)}

    def test_post_stim_limit(self):
        timing = {"pre_stim_s": 2.0, "total_s": 7.0, "fs": 1000}
        assert lfp_common.make_stft_params(
            timing, 3.0)["time_range_tuple"] == (0, 5.0)
        # never beyond the end of the epoch
        assert lfp_common.make_stft_params(
            timing, 10.0)["time_range_tuple"] == (0, 7.0)

    def test_window_scales_with_sampling_rate(self):
        timing = {"pre_stim_s": 1.0, "total_s": 4.0, "fs": 2000}
        params = lfp_common.make_stft_params(timing)
        assert params["signal_window"] == 1000
        assert params["window_overlap"] == 999


class TestNormalization:
    def test_baseline_is_zscored_per_trial_channel_frequency(self):
        rng = np.random.default_rng(0)
        time = np.arange(-2, 3, 0.1)
        power = rng.random((4, 2, 3, time.size)) + 5
        normalized = lfp_common.normalize_trial_power(power, time, 2.0)
        mask = lfp_common.baseline_window_mask(time, 2.0)
        assert np.allclose(normalized[..., mask].mean(axis=-1), 0)
        assert np.allclose(normalized[..., mask].std(axis=-1), 1)

    def test_baseline_stops_before_stimulus(self):
        time = np.arange(-2, 3, 0.05)
        mask = lfp_common.baseline_window_mask(time, 2.0)
        assert time[mask].max() <= -lfp_common.BASELINE_END_BUFFER_SECONDS
        assert time[mask].min() >= -2.0

    def test_short_pre_stimulus_period_raises(self):
        time = np.arange(-0.1, 3, 0.05)
        with pytest.raises(ValueError, match="baseline"):
            lfp_common.baseline_window_mask(time, 2.0)

    def test_summary_shapes(self):
        time = np.arange(-2, 3, 0.1)
        normalized = np.ones((4, 2, 3, time.size))
        summary = lfp_common.summarize_normalized_power(normalized, time)
        assert summary["spectrogram"].shape == (3, time.size)
        assert summary["pre_power"].shape == (3,)
        assert summary["post_power"].shape == (3,)


class TestTasteNames:
    @staticmethod
    def make_dat(tmp_path, node_names, dig_in_names, tastes):
        h5_path = str(tmp_path / "data.h5")
        with tables.open_file(h5_path, "w") as hf5:
            hf5.create_group("/", "Parsed_LFP")
            for name in node_names:
                hf5.create_array("/Parsed_LFP", name, np.zeros((2, 3, 4)))
        info = {"taste_params": {
            "dig_in_names": dig_in_names, "tastes": tastes}}

        def get_info_dict():
            dat.info_dict = info

        dat = SimpleNamespace(
            hdf5_path=h5_path, get_info_dict=get_info_dict,
            lfp_array_list=[None] * len(node_names))
        return dat

    def test_names_follow_dig_in_not_position(self, tmp_path):
        # HDF5 lists the nodes alphabetically (DIN_10_LFPs, DIN_1_LFPs,
        # DIN_2_LFPs), which differs from the dig-in order in the .info file
        dat = self.make_dat(
            tmp_path,
            ["DIN_1_LFPs", "DIN_10_LFPs", "DIN_2_LFPs"],
            ["DIN_1", "DIN_2", "DIN_10"], ["water", "salt", "sucrose"])
        assert lfp_common.get_taste_names_in_lfp_order(dat) == [
            "sucrose", "water", "salt"]

    def test_unknown_dig_in_raises(self, tmp_path):
        dat = self.make_dat(
            tmp_path, ["DIN_9_LFPs"], ["DIN_1"], ["water"])
        with pytest.raises(ValueError, match="no matching taste"):
            lfp_common.get_taste_names_in_lfp_order(dat)

    def test_count_mismatch_raises(self, tmp_path):
        dat = self.make_dat(
            tmp_path, ["DIN_1_LFPs"], ["DIN_1"], ["water"])
        dat.lfp_array_list = [None, None]
        with pytest.raises(ValueError, match="loaded LFP arrays"):
            lfp_common.get_taste_names_in_lfp_order(dat)


class TestLoadCleaned:
    @staticmethod
    def write_cleaned(data_dir, shapes, excluded=()):
        cleaned_dir = os.path.join(data_dir, lfp_common.CLEANED_LFP_FOLDER)
        os.makedirs(cleaned_dir)
        tastes = []
        for index, shape in enumerate(shapes):
            np.save(os.path.join(
                cleaned_dir, f"taste_{index:02d}_cleaned.npy"),
                np.zeros(shape))
            tastes.append({"taste_index": index, "output_shape": list(shape)})
        with open(os.path.join(
                cleaned_dir, "artifact_removal_report.json"), "w") as f:
            json.dump({"excluded_taste_indices": list(excluded),
                       "tastes": tastes}, f)

    def test_excluded_tastes_are_dropped(self, tmp_path):
        data_dir = str(tmp_path)
        self.write_cleaned(data_dir, [(2, 3, 4)] * 3, excluded=[1])
        dat = SimpleNamespace(lfp_array_list=[np.zeros((5, 3, 4))] * 3)
        lfp_common.load_cleaned_lfps(dat, data_dir, ["a", "b", "c"])
        assert dat.active_taste_names == ["a", "c"]
        assert dat.active_taste_indices.tolist() == [0, 2]
        assert len(dat.lfp_array_list) == 2

    def test_missing_report_raises(self, tmp_path):
        dat = SimpleNamespace(lfp_array_list=[np.zeros((5, 3, 4))])
        with pytest.raises(FileNotFoundError, match="remove_lfp_artifacts"):
            lfp_common.load_cleaned_lfps(dat, str(tmp_path), ["a"])

    def test_taste_count_mismatch_raises(self, tmp_path):
        data_dir = str(tmp_path)
        self.write_cleaned(data_dir, [(2, 3, 4)] * 2)
        dat = SimpleNamespace(lfp_array_list=[np.zeros((5, 3, 4))] * 3)
        with pytest.raises(ValueError, match="describes 2 tastes"):
            lfp_common.load_cleaned_lfps(dat, data_dir, ["a", "b", "c"])
