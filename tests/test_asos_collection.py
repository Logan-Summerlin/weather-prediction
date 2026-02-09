"""
Tests for ASOS collection helpers.
"""

from src.asos_collection import iter_date_chunks


def test_iter_date_chunks_single_year():
    chunks = iter_date_chunks("2020-01-01", "2020-12-31", chunk_years=1)
    assert chunks == [("2020-01-01", "2020-12-31")]


def test_iter_date_chunks_multiple_years():
    chunks = iter_date_chunks("2019-06-01", "2021-02-10", chunk_years=1)
    assert chunks == [
        ("2019-06-01", "2019-12-31"),
        ("2020-01-01", "2020-12-31"),
        ("2021-01-01", "2021-02-10"),
    ]


def test_iter_date_chunks_two_year_blocks():
    chunks = iter_date_chunks("2018-01-01", "2021-12-31", chunk_years=2)
    assert chunks == [
        ("2018-01-01", "2019-12-31"),
        ("2020-01-01", "2021-12-31"),
    ]
