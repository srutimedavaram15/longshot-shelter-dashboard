import pandas as pd
import numpy as np
import pytest

from dashboard import generate_reason, compute_majority_cols, get_missing_columns, REQUIRED_COLUMNS


# ── Helpers ────────────────────────────────────────────────────────────────────

def _row(active_cols, all_cols):
    data = {c: 0 for c in all_cols}
    for c in active_cols:
        data[c] = 1
    return pd.Series(data)


def _hr(mapping):
    return pd.Series(mapping)



# ── generate_reason ────────────────────────────────────────────────────────────

class TestGenerateReason:
    def test_two_non_majority_features_sorted_by_hazard(self):
        # Dog (HR=2.0) has larger |log HR| than Stray (HR=1.5), so Dog appears first.
        hr = _hr({"Animal Type_Dog": 2.0, "Intake Type_Stray": 1.5})
        row = _row(["Animal Type_Dog", "Intake Type_Stray"], list(hr.index))
        assert generate_reason(row, hr, majority_cols=set()) == (
            "Flagged primarily due to animal type (Dog) and intake type (Stray)."
        )

    def test_single_non_majority_feature(self):
        hr = _hr({"Intake Type_Stray": 1.5})
        row = _row(["Intake Type_Stray"], list(hr.index))
        assert generate_reason(row, hr, majority_cols=set()) == (
            "Flagged primarily due to intake type (Stray)."
        )

    def test_no_active_features_returns_baseline(self):
        hr = _hr({"Intake Type_Stray": 1.5, "Animal Type_Dog": 2.0})
        row = _row([], list(hr.index))
        assert generate_reason(row, hr, majority_cols=set()) == (
            "Profile matches baseline risk level."
        )

    def test_non_majority_sorted_before_majority(self):
        # Stray is marked majority; Dog is not — Dog should appear first.
        hr = _hr({"Intake Type_Stray": 1.5, "Animal Type_Dog": 2.0})
        row = _row(["Intake Type_Stray", "Animal Type_Dog"], list(hr.index))
        assert generate_reason(row, hr, majority_cols={"Intake Type_Stray"}) == (
            "Flagged primarily due to animal type (Dog) and intake type (Stray)."
        )

    def test_all_majority_features_still_produces_flagged_message(self):
        hr = _hr({"Intake Type_Stray": 1.5, "Animal Type_Dog": 2.0})
        row = _row(["Intake Type_Stray", "Animal Type_Dog"], list(hr.index))
        result = generate_reason(row, hr, majority_cols={"Intake Type_Stray", "Animal Type_Dog"})
        assert result.startswith("Flagged primarily due to")

    def test_unrecognized_feature_prefix_returns_baseline(self):
        # No column matches any FEATURE_READABLE key, so parts stays empty.
        hr = _hr({"UnknownGroup_foo": 2.0})
        row = _row(["UnknownGroup_foo"], list(hr.index))
        assert generate_reason(row, hr, majority_cols=set()) == (
            "Profile matches baseline risk level."
        )

    def test_only_two_features_used_even_when_more_active(self):
        hr = _hr({
            "Animal Type_Dog": 3.0,
            "Intake Type_Stray": 2.0,
            "Sex upon Intake_Male": 1.5,
        })
        row = _row(list(hr.index), list(hr.index))
        result = generate_reason(row, hr, majority_cols=set())
        # Result mentions exactly two features — the two with the highest |log HR|.
        assert result == (
            "Flagged primarily due to animal type (Dog) and intake type (Stray)."
        )


# ── compute_majority_cols ──────────────────────────────────────────────────────

class TestComputeMajorityCols:
    def test_single_feature_clear_majority(self):
        df = pd.DataFrame({
            "Intake Type_Stray": [1, 1, 1],
            "Intake Type_Owner Surrender": [0, 0, 1],
        })
        assert compute_majority_cols(df, ["Intake Type"]) == {"Intake Type_Stray"}

    def test_two_features_returns_majority_for_each(self):
        df = pd.DataFrame({
            "Intake Type_Stray": [1, 1, 0],
            "Intake Type_Owner Surrender": [0, 0, 1],
            "Animal Type_Dog": [1, 0, 0],
            "Animal Type_Cat": [0, 1, 1],
        })
        result = compute_majority_cols(df, ["Intake Type", "Animal Type"])
        assert result == {"Intake Type_Stray", "Animal Type_Cat"}

    def test_feature_with_no_matching_columns_is_skipped(self):
        df = pd.DataFrame({
            "Animal Type_Dog": [1, 1, 0],
            "Animal Type_Cat": [0, 0, 1],
        })
        # "Intake Type" has no columns in df — it is silently skipped.
        result = compute_majority_cols(df, ["Intake Type", "Animal Type"])
        assert result == {"Animal Type_Dog"}

    def test_empty_dataframe_returns_first_column_per_group(self):
        # All sums are 0; idxmax on a zero Series returns the first column.
        df = pd.DataFrame({"Intake Type_Stray": pd.Series([], dtype=int),
                           "Intake Type_Owner Surrender": pd.Series([], dtype=int)})
        result = compute_majority_cols(df, ["Intake Type"])
        assert result == {"Intake Type_Stray"}


# ── Column validation ──────────────────────────────────────────────────────────

class TestColumnValidation:
    def test_all_required_columns_plus_name_is_valid(self):
        assert get_missing_columns(REQUIRED_COLUMNS + ["Name"], REQUIRED_COLUMNS) == []

    def test_animal_id_satisfies_name_requirement(self):
        assert get_missing_columns(REQUIRED_COLUMNS + ["Animal ID"], REQUIRED_COLUMNS) == []

    def test_missing_all_columns_reports_everything(self):
        missing = get_missing_columns([], REQUIRED_COLUMNS)
        assert set(missing) == set(REQUIRED_COLUMNS) | {"Name (or Animal ID)"}
        assert len(missing) == len(REQUIRED_COLUMNS) + 1

    def test_partially_missing_columns(self):
        present = ["Animal Type", "Age upon Intake", "Intake Type", "Name"]
        missing = get_missing_columns(present, REQUIRED_COLUMNS)
        assert "Intake Condition" in missing
        assert "Sex upon Intake" in missing
        assert "intake_date" in missing
        assert "Animal Type" not in missing
        assert "Name (or Animal ID)" not in missing

    def test_all_required_but_no_name_column(self):
        # Has all data columns but neither Name nor Animal ID.
        assert get_missing_columns(REQUIRED_COLUMNS, REQUIRED_COLUMNS) == ["Name (or Animal ID)"]

    def test_extra_columns_do_not_affect_validation(self):
        headers = REQUIRED_COLUMNS + ["Name", "Breed", "Color", "Zip Code"]
        assert get_missing_columns(headers, REQUIRED_COLUMNS) == []
