import pytest

from cleancut.edl_ops import find_overlapping_shot, fmt_timestamp, parse_timestamp
from cleancut.scenes import Shot


@pytest.mark.parametrize("s,expected", [
    ("0", 0.0),
    ("5", 5.0),
    ("5.25", 5.25),
    ("1:00", 60.0),
    ("01:30", 90.0),
    ("00:30.5", 30.5),
    ("1:02:03", 3723.0),
    ("1:02:03.5", 3723.5),
])
def test_parse_timestamp_valid(s, expected):
    assert parse_timestamp(s) == expected


@pytest.mark.parametrize("s", ["", "abc", "1:2:3:4", "1:xx", ":30"])
def test_parse_timestamp_invalid(s):
    with pytest.raises(ValueError):
        parse_timestamp(s)


def test_fmt_timestamp_under_hour():
    assert fmt_timestamp(75.5) == "1:15.50"


def test_fmt_timestamp_over_hour():
    assert fmt_timestamp(3723.5) == "1:02:03.50"


def test_find_overlapping_shot_with_objects():
    shots = [Shot(0, 10), Shot(10, 20), Shot(20, 30)]
    assert find_overlapping_shot(5, shots) == (0.0, 10.0)
    assert find_overlapping_shot(10, shots) == (10.0, 20.0)
    assert find_overlapping_shot(99, shots) is None


def test_find_overlapping_shot_with_tuples():
    shots = [(0, 10), (10, 20)]
    assert find_overlapping_shot(15, shots) == (10.0, 20.0)
