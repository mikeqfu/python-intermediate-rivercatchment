"""Tests for statistics functions within the Model layer."""

import datetime

import numpy as np
import numpy.testing as npt
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


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (
                pd.DataFrame(
                    data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    index=[pd.to_datetime('2000-01-01 01:00'),
                           pd.to_datetime('2000-01-01 02:00'),
                           pd.to_datetime('2000-01-01 03:00')],
                    columns=['A', 'B', 'C']
                ),
                pd.DataFrame(
                    data=[[0.0, 0.0, 0.0]],
                    index=[datetime.date(2000, 1, 1)],
                    columns=['A', 'B', 'C']
                )
        ),
        (
                pd.DataFrame(
                    data=[[4, 2, 5], [1, 6, 2], [4, 1, 9]],
                    index=[pd.to_datetime('2000-01-01 01:00'),
                           pd.to_datetime('2000-01-01 02:00'),
                           pd.to_datetime('2000-01-01 03:00')],
                    columns=['A', 'B', 'C']
                ),
                pd.DataFrame(
                    data=[[4, 6, 9]],
                    index=[datetime.date(2000, 1, 1)],
                    columns=['A', 'B', 'C']
                )
        ),
        (
                pd.DataFrame(
                    data=[[4, -2, 5], [1, -6, 2], [-4, -1, 9]],
                    index=[pd.to_datetime('2000-01-01 01:00'),
                           pd.to_datetime('2000-01-01 02:00'),
                           pd.to_datetime('2000-01-01 03:00')],
                    columns=['A', 'B', 'C']
                ),
                pd.DataFrame(
                    data=[[4, -1, 9]],
                    index=[datetime.date(2000, 1, 1)],
                    columns=['A', 'B', 'C']
                )
        ),
    ])
def test_daily_max(test_input, expected_output):
    """Test max function works for array of zeroes and positive integers."""
    from catchment.models import daily_max

    pdt.assert_frame_equal(daily_max(test_input), expected_output)


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        (
                pd.DataFrame(
                    data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    index=[pd.to_datetime('2000-01-01 01:00'),
                           pd.to_datetime('2000-01-01 02:00'),
                           pd.to_datetime('2000-01-01 03:00')],
                    columns=['A', 'B', 'C']
                ),
                pd.DataFrame(
                    data=[[0.0, 0.0, 0.0]],
                    index=[datetime.date(2000, 1, 1)],
                    columns=['A', 'B', 'C']
                )
        ),
        (
                pd.DataFrame(
                    data=[[4, 2, 5], [1, 6, 2], [4, 1, 9]],
                    index=[pd.to_datetime('2000-01-01 01:00'),
                           pd.to_datetime('2000-01-01 02:00'),
                           pd.to_datetime('2000-01-01 03:00')],
                    columns=['A', 'B', 'C']
                ),
                pd.DataFrame(
                    data=[[1, 1, 2]],
                    index=[datetime.date(2000, 1, 1)],
                    columns=['A', 'B', 'C']
                )
        ),
        (
                pd.DataFrame(
                    data=[[4, -2, 5], [1, -6, 2], [-4, -1, 9]],
                    index=[pd.to_datetime('2000-01-01 01:00'),
                           pd.to_datetime('2000-01-01 02:00'),
                           pd.to_datetime('2000-01-01 03:00')],
                    columns=['A', 'B', 'C']
                ),
                pd.DataFrame(
                    data=[[-4, -6, 2]],
                    index=[datetime.date(2000, 1, 1)],
                    columns=['A', 'B', 'C']
                )
        ),
    ])
def test_daily_min(test_input, expected_output):
    """Test min function works for array of zeroes and positive integers."""
    from catchment.models import daily_min

    pdt.assert_frame_equal(daily_min(test_input), expected_output)


def test_daily_min_python_list():
    """Test for AttributeError when passing a python list"""
    from catchment.models import daily_min

    with pytest.raises(AttributeError):
        _ = daily_min([[3, 4, 7], [-3, 0, 5]])


@pytest.mark.parametrize(
    "test_data, test_index, test_columns, expected_data, expected_index, expected_columns",
    [
        (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [pd.to_datetime('2000-01-01 01:00'),
                 pd.to_datetime('2000-01-01 02:00'),
                 pd.to_datetime('2000-01-01 03:00')],
                ['A', 'B', 'C'],
                [[0.14, 0.25, 0.33], [0.57, 0.63, 0.66], [1.0, 1.0, 1.0]],
                [pd.to_datetime('2000-01-01 01:00'),
                 pd.to_datetime('2000-01-01 02:00'),
                 pd.to_datetime('2000-01-01 03:00')],
                ['A', 'B', 'C']
        ),
    ])
def test_normalise(test_data, test_index, test_columns, expected_data, expected_index,
                   expected_columns):
    """Test normalisation works for arrays of one and positive integers.
       Assumption that test accuracy of two decimal places is sufficient."""
    from catchment.models import data_normalise

    a = data_normalise(pd.DataFrame(data=test_data, index=test_index, columns=test_columns))
    b = pd.DataFrame(data=expected_data, index=expected_index, columns=expected_columns)
    pdt.assert_frame_equal(a, b, atol=1e-2)


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        # previous test cases here, with None for expect_raises, except for the next one -
        #   add ValueError as an expected exception (since it has a negative input value)
        (
                [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
                ValueError,
        ),
        (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[0.14, 0.25, 0.33], [0.57, 0.63, 0.67], [1.0, 1.0, 1.0]],
                None,
        ),
        (
                'hello',
                None,
                TypeError,
        ),
        (
                3,
                None,
                TypeError,
        ),
    ])
def test_data_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""
    from catchment.models import data_normalise

    if isinstance(test, list):
        test = np.array(test)

    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_almost_equal(data_normalise(test), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(data_normalise(test), np.array(expected), decimal=2)


def test_create_site():
    """Check a site is created correctly given a name."""
    from catchment.models import Site

    name = 'PL23'
    p = Site(name=name)
    assert p.name == name


def test_create_catchment():
    """Check a catchment is created correctly given a name."""
    from catchment.models import Catchment

    name = 'Spain'
    catchment = Catchment(name=name)
    assert catchment.name == name


def test_catchment_is_location():
    """Check if a catchment is a location."""
    from catchment.models import Catchment, Location

    catchment = Catchment("Spain")
    assert isinstance(catchment, Location)


def test_site_is_location():
    """Check if a site is a location."""
    from catchment.models import Site, Location

    PL23 = Site("PL23")
    assert isinstance(PL23, Location)


def test_sites_added_correctly():
    """Check sites are being added correctly by a catchment. """
    from catchment.models import Catchment, Site

    catchment = Catchment("Spain")
    PL23 = Site("PL23")
    catchment.add_site(PL23)
    assert catchment.sites is not None
    assert len(catchment.sites) == 1


def test_no_duplicate_sites():
    """Check adding the same site to the same catchment twice does not result in duplicates. """
    from catchment.models import Catchment, Site

    catchment = Catchment("Sheila Wheels")
    PL23 = Site("PL23")
    catchment.add_site(PL23)
    catchment.add_site(PL23)
    assert len(catchment.sites) == 1


if __name__ == '__main__':
    pytest.main()
