import math

from scipy import stats
import numpy as np
import statsmodels.api as sm


# values_type = 'c' + 'u' + 'c' + 'c' + 'c' + 'c' + 'c' + 'c' + 'o' + 'o' + 'o' + 'o' + 'uu' + 'uu' + 'uuu' + 'uuuuu' + 'uuuu' + 'uuuuu' + 'uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu' + 'uuuuuu' + 'uuuuu' + 'uu' + 'uu' + 'uu' + 'uuuu' + 'uu'


def distribution_normalization(input):
    data_std = np.std(input, axis=0)
    data_mean = np.mean(input, axis=0)
    norm = (input - data_mean) / (data_std + 1e-4) + 1e-4
    return norm


def distribution_modifier(data_set):
    data_sum_min = np.min(data_set[0], axis=0)
    data_sum_max = np.max(data_set[0], axis=0)
    # percentage_x = (data_set[0] - data_sum_min) / (data_sum_max - data_sum_min+ 1e-4)
    norm_x = distribution_normalization(data_set[0])
    # kernel = stats.gaussian_kde(np.unique(integrated_data.T, axis=1), bw_method=1)
    dens = sm.nonparametric.KDEMultivariate(norm_x, var_type='c' * norm_x.shape[1], bw=np.repeat(0.5, norm_x.shape[1]))

    return dens


"""
data: shall be in size of (n_data_point,n_features)
"""


def kernel_function(data):
    std = np.std(data, axis=0)
    avg = np.average(data, axis=0)
    kernel_out = (1 / np.sqrt(2 * np.pi * np.square(std))) * np.power(np.e, -np.square(data - avg) / 2 * np.square(std))
    return kernel_out


def my_kde(data, h=1):
    kernel = kernel_function(data)
    
    return
