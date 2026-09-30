import pandas as pd
from pandas.testing import assert_frame_equal

from mnq_signal_lab.data import generate_synthetic
from mnq_signal_lab.features import FEATURE_COLUMNS, add_labels, make_features


def test_features_do_not_change_when_future_is_appended():
    raw = generate_synthetic("2024-01-02", periods=500)
    prefix = make_features(raw.iloc[:350])
    full = make_features(raw)
    assert_frame_equal(prefix[FEATURE_COLUMNS], full.iloc[:350][FEATURE_COLUMNS])


def test_labels_point_strictly_forward():
    frame = add_labels(make_features(generate_synthetic("2024-01-02", 300)), 5, 2, 0.25)
    valid = frame.dropna(subset=["label_end_time"])
    assert (valid.label_end_time > valid.timestamp).all()
    assert pd.api.types.is_datetime64_any_dtype(valid.label_end_time)
