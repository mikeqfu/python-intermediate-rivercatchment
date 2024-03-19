"""Tests for statistics functions within the Model layer."""

import datetime

import pandas as pd
import pandas.testing as pdt
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from catchment.models import daily_mean

    test_input = pd.DataFrame(
        data=[[0.0, 0.0],
              [0.0, 0.0],
              [0.0, 0.0]],
        index=[pd.to_datetime('2000-01-01 01:00'),
               pd.to_datetime('2000-01-01 02:00'),
               pd.to_datetime('2000-01-01 03:00')],
        columns=['A', 'B']
    )
    test_result = pd.DataFrame(
        data=[[0.0, 0.0]],
        index=[datetime.date(2000, 1, 1)],
        columns=['A', 'B']
    )

    # Need to use Pandas testing functions to compare arrays
    pdt.assert_frame_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from catchment.models import daily_mean

    test_input = pd.DataFrame(
        data=[[1, 2],
              [3, 4],
              [5, 6]],
        index=[pd.to_datetime('2000-01-01 01:00'),
               pd.to_datetime('2000-01-01 02:00'),
               pd.to_datetime('2000-01-01 03:00')],
        columns=['A', 'B']
    )
    test_result = pd.DataFrame(
        data=[[3.0, 4.0]],
        index=[datetime.date(2000, 1, 1)],
        columns=['A', 'B']
    )

    # Need to use Pandas testing functions to compare arrays
    pdt.assert_frame_equal(daily_mean(test_input), test_result)


def test_daily_max():
    """Test that max function works for an array of positive integers."""
    from catchment.models import daily_max

    test_input = pd.DataFrame(
        data=[[4, 2, 5],
              [1, 6, 2],
              [4, 1, 9]],
        index=[pd.to_datetime('2000-01-01 01:00'),
               pd.to_datetime('2000-01-01 02:00'),
               pd.to_datetime('2000-01-01 03:00')],
        columns=['A', 'B', 'C']
    )
    test_result = pd.DataFrame(
        data=[[4, 6, 9]],
        index=[datetime.date(2000, 1, 1)],
        columns=['A', 'B', 'C']
    )

    pdt.assert_frame_equal(daily_max(test_input), test_result)


def test_daily_min():
    """Test that min function works for an array of positive and negative integers."""
    from catchment.models import daily_min

    test_input = pd.DataFrame(
        data=[[4, -2, 5],
              [1, -6, 2],
              [-4, -1, 9]],
        index=[pd.to_datetime('2000-01-01 01:00'),
               pd.to_datetime('2000-01-01 02:00'),
               pd.to_datetime('2000-01-01 03:00')],
        columns=['A', 'B', 'C']
    )
    test_result = pd.DataFrame(
        data=[[-4, -6, 2]],
        index=[datetime.date(2000, 1, 1)],
        columns=['A', 'B', 'C']
    )

    pdt.assert_frame_equal(daily_min(test_input), test_result)


def test_daily_min_python_list():
    """Test for AttributeError when passing a python list"""
    from catchment.models import daily_min

    with pytest.raises(AttributeError):
        _ = daily_min([[3, 4, 7], [-3, 0, 5]])


if __name__ == '__main__':
    pytest.main()
