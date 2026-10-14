"""Tests for the taste-exclusion handling of remove_lfp_artifacts.py."""
import pytest

import remove_lfp_artifacts as rla

TASTES = ["Water", "NaCl", "Sucrose", "Quinine", "NaCl"]


class TestResolveTasteExclusions:
    def test_by_name(self):
        assert rla.resolve_taste_exclusions(["Quinine"], TASTES) == {3}

    def test_name_ignoring_case(self):
        assert rla.resolve_taste_exclusions(["quinine"], TASTES) == {3}

    def test_duplicate_taste_names_are_all_excluded(self):
        assert rla.resolve_taste_exclusions(["NaCl"], TASTES) == {1, 4}

    def test_by_index_and_file_style_index(self):
        assert rla.resolve_taste_exclusions(["0", "taste_02"], TASTES) == {0, 2}

    def test_name_wins_over_index(self):
        assert rla.resolve_taste_exclusions(["1"], ["a", "1", "b"]) == {1}

    def test_unknown_name_raises_with_taste_list(self):
        with pytest.raises(ValueError, match="Quinnine.*Sucrose"):
            rla.resolve_taste_exclusions(["Quinnine"], TASTES)

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="valid index"):
            rla.resolve_taste_exclusions(["9"], TASTES)


class TestParseExclusions:
    def test_groups_by_dataset(self):
        parsed = rla.parse_exclusions(["day1:Quinine", "day1:NaCl", "day2:0"])
        assert parsed == {"day1": {"Quinine", "NaCl"}, "day2": {"0"}}

    @pytest.mark.parametrize("value", ["Quinine", "day1:", ":Quinine"])
    def test_malformed_values_exit(self, value):
        with pytest.raises(SystemExit):
            rla.parse_exclusions([value])
