import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


def read_from_json(filename):
    with open(filename, 'r') as f:
        result = json.load(f)
        return result


def plot_density_hist(json_file):
    base_auc = json_file['all year AUC']
    aug_auc = json_file['augmentation AUC']
    dims_list = []
    plt.plot(dims_list, base_auc,'-', label='Baseline AUC')
    plt.plot(dims_list, aug_auc,'-', label='Augmentation AUC')
    plt.legend()
    plt.title('Performance of KDE with different dimensionality')
    plt.ylabel('AUC of testing set')
    plt.xlabel('Number of Dimensions')
    # plt.show()


def plot_bandwidth(json_file, num_dimensions):
    base_auc = json_file['all year AUC']
    aug_auc = json_file['augmentation AUC']
    bandwidth = json_file['bandwidth']
    plt.plot(bandwidth, base_auc, '-',label='Baseline AUC')
    plt.plot(bandwidth, aug_auc, '-',label='PNKDEAC')
    plt.legend()
    plt.title('Performance of KDE with different bandwidth an PCA in {} dimensions'.format(num_dimensions))
    plt.ylabel('AUC of testing set')
    plt.xlabel('Bandwidth in Naive KDE')
    plt.show()


def plot_bandwidth_year(json_file, num_dimensions, year):
    base_auc = json_file['all year AUC']
    aug_auc = json_file['augmentation AUC']
    bandwidth = json_file['bandwidth']
    plt.plot(bandwidth, base_auc, '-',label='Baseline AUC')
    plt.plot(bandwidth, aug_auc, '-',label='Augmentation AUC')
    plt.legend()
    plt.title('Performance of KDE with different bandwidth an PCA \n in {} dimensions in {}'.format(num_dimensions, year))
    plt.ylabel('AUC of testing set')
    plt.xlabel('Bandwidth in Naive KDE')
    plt.show()

def plot_all_year(json_file, num_dimensions):
    years = list(range(2006, 2021))
    limited_years = [*[2006],*list(range(2009, 2021))]
    plt.plot(years, json_file['all year AUC'],'-', label="Baseline Model")
    plt.plot(years, json_file['augmentation AUC'],'-', label="Augmentation Model")
    plt.title("AUC of Gradient Boosting with PCA in {} dimensions".format(num_dimensions))
    plt.xlabel("Year")
    plt.ylabel("AUC")
    plt.legend()
    plt.show()
