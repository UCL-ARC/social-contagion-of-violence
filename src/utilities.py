import os
import matplotlib.pyplot as plt
import numpy as np


def dict_string(d):
    return str(d).replace("{", "").replace("}", "").replace("'", "").replace(":", "=")


def set_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        for f in os.listdir(dir_name):
            os.unlink(os.path.join(dir_name,f))
    return dir_name


def round_to_n(value, n):
    return str('{:g}'.format(float('{:.{p}g}'.format(value, p=n))))


def top_n(values, percent=1):
    highest_risk = np.zeros(len(values))
    n = int(np.ceil(len(values) * percent / 100))
    value = -np.sort(-values)[n - 1]
    highest_risk[values >= value] = 1
    return highest_risk


def norm(arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(arr, np.sum(arr, 1)[:, np.newaxis])
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c


def enhance_plot(fig=None, show=True, filename=None, params_dict=None, dir_name='results', clear=True):
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
    if clear:
        fig.clf()
