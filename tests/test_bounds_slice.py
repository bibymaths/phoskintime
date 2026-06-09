import pytest
from networkmodel.OptimizationProblem import _validate_slice_layout_covers_bounds


def test_slice_layout_valid_contiguous_with_empty_group():
    slices = {
        "a": slice(0, 2),
        "empty": slice(2, 2),
        "b": slice(2, 5),
    }

    assert _validate_slice_layout_covers_bounds(slices, 5) == 5


def test_slice_layout_allows_empty_slice_at_end():
    slices = {
        "a": slice(0, 5),
        "empty": slice(5, 5),
    }

    assert _validate_slice_layout_covers_bounds(slices, 5) == 5


def test_slice_layout_allows_empty_slice_at_start():
    slices = {
        "empty": slice(0, 0),
        "a": slice(0, 5),
    }

    assert _validate_slice_layout_covers_bounds(slices, 5) == 5


def test_slice_layout_rejects_gap():
    slices = {
        "a": slice(0, 2),
        "b": slice(3, 5),
    }

    with pytest.raises(ValueError, match="gap"):
        _validate_slice_layout_covers_bounds(slices, 5)


def test_slice_layout_rejects_overlap():
    slices = {
        "a": slice(0, 3),
        "b": slice(2, 5),
    }

    with pytest.raises(ValueError, match="overlaps"):
        _validate_slice_layout_covers_bounds(slices, 5)


def test_slice_layout_rejects_missing_start():
    slices = {
        "a": slice(1, 5),
    }

    with pytest.raises(ValueError, match="start at 0|does not cover bounds index 0"):
        _validate_slice_layout_covers_bounds(slices, 5)


def test_slice_layout_rejects_missing_tail():
    slices = {
        "a": slice(0, 4),
    }

    with pytest.raises(ValueError, match="missing tail"):
        _validate_slice_layout_covers_bounds(slices, 5)


def test_slice_layout_rejects_exceeding_bounds():
    slices = {
        "a": slice(0, 6),
    }

    with pytest.raises(ValueError, match="exceeds bounds"):
        _validate_slice_layout_covers_bounds(slices, 5)


def test_slice_layout_rejects_reverse_slice():
    slices = {
        "a": slice(3, 2),
    }

    with pytest.raises(ValueError, match="invalid bounds"):
        _validate_slice_layout_covers_bounds(slices, 5)


def test_slice_layout_rejects_non_unit_step():
    slices = {
        "a": slice(0, 5, 2),
    }

    with pytest.raises(ValueError, match="step"):
        _validate_slice_layout_covers_bounds(slices, 5)