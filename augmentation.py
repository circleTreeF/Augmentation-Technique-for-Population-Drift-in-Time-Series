import math

import self as self
from scipy import stats
import numpy as np
import statsmodels.api as sm
from KDEpy import NaiveKDE, TreeKDE, FFTKDE


# values_type = 'c' + 'u' + 'c' + 'c' + 'c' + 'c' + 'c' + 'c' + 'o' + 'o' + 'o' + 'o' + 'uu' + 'uu' + 'uuu' + 'uuuuu' + 'uuuu' + 'uuuuu' + 'uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu' + 'uuuuuu' + 'uuuuu' + 'uu' + 'uu' + 'uu' + 'uuuu' + 'uu'


def distribution_normalization(input):
    data_std = np.std(input, axis=0)
    data_mean = np.mean(input, axis=0)
    norm = (input - data_mean) / (data_std + 1e-4) + 1e-4
    return norm


def distribution_modifier(data_set):
    data_sum_min = np.min(data_set, axis=0)
    data_sum_max = np.max(data_set, axis=0)
    # percentage_x = (data_set[0] - data_sum_min) / (data_sum_max - data_sum_min+ 1e-4)
    norm_x = distribution_normalization(data_set)
    # kernel = stats.gaussian_kde(np.unique(integrated_data.T, axis=1), bw_method=1)
    # KDE Method 1
    dens = sm.nonparametric.KDEMultivariate(norm_x, var_type='c' * norm_x.shape[1], bw=np.repeat(0.4, norm_x.shape[1]))
    # KDE Method 2
    naive_kde = TreeKDE(kernel='gaussian', bw=0.4).fit(norm_x)
    return dens, naive_kde


"""
My implementation of KDE
    # KDE Method 3 
data: shall be in size of (n_data_point,n_features)
References:
Silverman, B.W. (1998). Density Estimation for Statistics and Data Analysis (1st ed.). Routledge. https://doi.org/10.1201/9781315140919
"""


class KDE:
    """
    __sample: in shape of [n_data_point, n_features]
    """

    def __init__(self, __sample, __h=1):
        self.sample = __sample
        self.h = __h

    def gaussian_kernel(self, input):
        # (1. / np.sqrt(2 * np.pi)) * np.exp(-(Xi - x) ** 2 / (h ** 2 * 2.))
        # K(\mathbf{x})=(2 \pi)^{-d / 2} \exp \left(-\frac{1}{2} \mathbf{x}^{\mathbf{T}} \mathbf{x}\right)
        kernel_out = (1 / np.power(2 * np.pi, self.sample.shape[1] / 2)) * np.exp(-np.matmul(input, input.T) / 2)
        return kernel_out

    def my_kde(self, data_point):
        kernel = 0.
        for X_i in self.sample:
            kernel += self.gaussian_kernel((data_point - X_i) / self.h)
        return kernel / (self.sample.shape[0] * np.power(self.h, self.sample.shape[1]))


"""
# calculate the pdf for each data point
for idx, (i) in enumerate(x):
    my_pdf[idx]=kde.my_kde(i)
    
"""
