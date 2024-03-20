"""Module containing models representing catchment data.

The Model layer is responsible for the 'business logic' part of the software.

Catchment data is held in a Pandas dataframe (2D array) where each column contains
data for a single measurement site, and each row represents a single measurement
time across all sites.
"""

import functools

import numpy as np
import pandas as pd


def read_variable_from_csv(filename):
    """Reads a named variable from a CSV file, and returns a
    pandas dataframe containing that variable. The CSV file must contain
    a column of dates, a column of site ID's, and (one or more) columns
    of data - only one of which will be read.

    :param filename: Filename of CSV to load
    :return: 2D array of given variable. Index will be dates,
             Columns will be the individual sites
    """
    dataset = pd.read_csv(filename, usecols=['Date', 'Site', 'Rainfall (mm)'])

    dataset = dataset.rename({'Date': 'OldDate'}, axis='columns')
    dataset['Date'] = [pd.to_datetime(x, dayfirst=True) for x in dataset['OldDate']]
    dataset = dataset.drop('OldDate', axis='columns')

    newdataset = pd.DataFrame(index=dataset['Date'].unique())

    for site in dataset['Site'].unique():
        newdataset[site] = dataset[dataset['Site'] == site].set_index('Date')["Rainfall (mm)"]

    newdataset = newdataset.sort_index()

    return newdataset


def daily_total(data):
    """Calculate the daily total of a 2D data array.

    :param data: A 2D Pandas data frame with measurement data.
                 Index must be np.datetime64 compatible format.
                 Columns are measurement sites.
    :returns: A 2D Pandas data frame with minimum values of the measurements for each day.
    """
    return data.groupby(data.index.date).sum()


def daily_mean(data):
    """Calculate the daily mean of a 2D data array.

    :param data: A 2D Pandas data frame with measurement data. 
                 Index must be np.datetime64 compatible format. 
                 Columns are measurement sites.
    :returns: A 2D Pandas data frame with maximum values of the measurements for each day.
    """
    return data.groupby(data.index.date).mean()


def daily_max(data):
    """Calculate the daily max of a 2D data array.

    :param data: A 2D Pandas data frame with measurement data. 
                 Index must be np.datetime64 compatible format. 
                 Columns are measurement sites.
    :returns: A 2D Pandas data frame with maximum values of the measurements for each day.
    """
    return data.groupby(data.index.date).max()


def daily_min(data):
    """Calculate the daily min of a 2D data array.
    
    :param data: A 2D Pandas data frame with measurement data. 
                 Index must be np.datetime64 compatible format. 
                 Columns are measurement sites.
    :returns: A 2D Pandas data frame with maximum values of the measurements for each day.   
    """
    return data.groupby(data.index.date).min()


def data_normalise(data):
    """
    Normalise any given 2D data array

    NaN values are replaced with a value of 0

    :param data: 2D array of inflammation data
    :type data: ndarray
    """
    if not isinstance(data, np.ndarray) and not isinstance(data, pd.DataFrame):
        raise TypeError('data input should be DataFrame or ndarray')
    if len(data.shape) != 2:
        raise ValueError('data array should be 2-dimensional')
    if np.any(data < 0):
        raise ValueError('Measurement values should be non-negative')

    data_ = np.nanmax(data, axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / data_[np.newaxis, :]

    normalised[np.isnan(normalised)] = 0

    return normalised


def daily_above_threshold(site_id, data, threshold):
    """Determine whether each data value exceeds a given threshold for a given site.

    :param site_id: The identifier for the site column
    :param data: A 2D Pandas data frame with measurement data. Columns are measurement sites.
    :param threshold: A threshold value to check against
    :returns: A boolean list representing whether each data point for a given site exceeded
        the threshold
    """

    return list(map(lambda x: x > threshold, data[site_id]))


def data_above_threshold(site_id, data, threshold):
    """Count how many data points for a given site exceed a given threshold.

    :param site_id: The identifier for the site column
    :param data: A 2D Pandas data frame with measurement data. Columns are measurement sites.
    :param threshold: A threshold value to check against
    :returns: An integer representing the number of data points over a given threshold
   """

    above_threshold = map(lambda x: x > threshold, data[site_id])
    return functools.reduce(lambda a, b: a + 1 if b else a, above_threshold, 0)


class MeasurementSeries:
    def __init__(self, series, name, units):
        self.series = series
        self.name = name
        self.units = units
        self.series.name = self.name

    def add_measurement(self, data):
        self.series = pd.concat([self.series, data])
        self.series.name = self.name

    def __str__(self):
        if self.units:
            return f"{self.name} ({self.units})"
        else:
            return self.name


class Location:
    """A Location."""
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Site(Location):
    """A measurement site in the study."""

    def __init__(self, name):
        super().__init__(name)
        self.measurements = {}


def add_measurement(self, measurement_id, data, units=None):
    if measurement_id in self.measurements.keys():
        self.measurements[measurement_id].add_measurement(data)

    else:
        self.measurements[measurement_id] = MeasurementSeries(data, measurement_id, units)


@property
def last_measurements(self):
    return pd.concat(
        [self.measurements[key].series[-1:] for key in self.measurements.keys()],
        axis=1).sort_index()


class Catchment(Location):
    """A catchment area in the study."""

    def __init__(self, name):
        super().__init__(name)
        self.sites = {}

    def add_site(self, new_site):
        # Basic check to see if the site has already been added to the catchment area
        for site in self.sites:
            if site == new_site:
                print(f'{new_site} has already been added to site list')
                return

        self.sites[new_site.name] = Site(new_site)
