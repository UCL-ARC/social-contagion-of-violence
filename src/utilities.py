import os
import matplotlib.pyplot as plt
import numpy as np


def dict_string(d):
    return str(d).replace("{", "").replace("}", "").replace("'", "").replace(":", "=")


def set_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def round_to_n(value, n):
    return str('{:g}'.format(float('{:.{p}g}'.format(value, p=n))))


def enhance_plot(fig=None, show=True, filename=None, params_dict=None, dir_name='results'):
    if fig is None:
        fig = plt.gcf()
    if params_dict is not None:
        params_string = dict_string(params_dict)
        fig.text(0.01, 0.01, f'Parameters: {params_string}', fontsize=10, wrap=True)
        fig.subplots_adjust(bottom=0.2)
    if filename is not None:
        fig.savefig(os.path.join(dir_name, filename))
    if show:
        fig.show()


def plot_mean_median(ax, data):
    ax.axvline(x=np.nanmean(data), linewidth=2, color='r', label=f'mean: {round_to_n(np.average(data), 3)}')
    ax.axvline(x=np.nanmedian(data), linewidth=2, color='g', label=f'median: {round_to_n(np.median(data), 3)}')
