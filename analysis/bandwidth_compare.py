import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


def read_from_json(filename):
    with open(filename, 'r') as f:
        result = json.load(f)
        return result


def plot_perf(json_file):
    base_auc = json_file['all year AUC']
    aug_auc = json_file['augmentation AUC']
    bandwidth = json_file['bandwidth']
    plt.plot(bandwidth, base_auc, '-', label='Baseline AUC')
    plt.plot(bandwidth, aug_auc, '-', label='Augmentation AUC')
    plt.legend()
    plt.title('Performance of KDE with different bandwidth')
    plt.ylabel('AUC of testing set')
    plt.xlabel('Bandwidth in Naive KDE')
    plt.show()


def plot_perf_adjusted(json_file):
    base_auc = json_file['all year AUC'][:4] + json_file['all year AUC'][15:] + json_file['all year AUC'][4:15]
    aug_auc = json_file['augmentation AUC'][:4] + json_file['augmentation AUC'][15:] + json_file['augmentation AUC'][
                                                                                       4:15]
    bandwidth = json_file['bandwidth'][:4] + json_file['bandwidth'][15:] + json_file['bandwidth'][4:15]
    plt.plot(bandwidth, base_auc, '-', label='Baseline AUC')
    plt.plot(bandwidth, aug_auc, '-', label='Augmentation AUC')
    plt.legend()
    plt.title('Performance of KDE with different bandwidth')
    plt.ylabel('AUC of testing set')
    plt.xlabel('Bandwidth in Native KDE')
    plt.show()


def ndarray_to_csv(array, path):
    pd.DataFrame(array).to_csv(path)
