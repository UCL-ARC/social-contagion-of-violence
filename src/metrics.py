import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
import src.utilities as ut


def _top_percent(arr, percent=1):
    highest_risk = np.zeros(len(arr))
    n = int(np.ceil(len(arr) * percent / 100))
    highest_risk[np.argpartition(arr, -n)[-n:]] = 1
    return highest_risk


def confusion_matrix(y_true, risk, percentages, labels=(0, 1)):
    if y_true.shape != risk.shape:
        raise ValueError('y_true and risk should have the same shape')
    cm_time = np.zeros([len(percentages), 4], dtype=int)
    for i, percent in enumerate(percentages):
        for y_true_time, risk_time in zip(y_true, risk):
            y_pred_time = _top_percent(risk_time, percent)
            cm_time[i] += cm(y_true_time, y_pred_time, labels=labels).flatten()
    return cm_time


def plot_cdf(cms, percentages, time, show=True, filename=None, directory='results'):
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # Main plot
    ax2 = ax1.twinx().twiny()
    for data_name, data_values in cms.items():
        ax1.plot(percentages / 100, data_values[:, 3] / np.sum(data_values[:, 2:], 1), '-o', label=data_name)
        ax2.scatter(np.sum(data_values[:, [1, 3]], 1) / time, data_values[:, 3], label=data_name, )
    ax1.set_title('Cumulative Distribution Function')
    ax1.set_xlabel('Proportion of nodes selected')
    ax1.set_ylabel('Hit Rate / Proportion of nodes identified')
    ax2.set_xlabel('Number of nodes selected per time unit')
    # TODO investigate why this ylabel does not appear
    ax2.set_ylabel('Number of nodes identified')
    ax2.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 36))

    # Inset
    axins = ax1.inset_axes([0.6, 0.07, 0.37, 0.47])
    for data_name, data_values in cms.items():
        axins.plot(percentages / 100, data_values[:, 3] / np.sum(data_values[:, 2:], 1), '-o')
    x1, x2, y1, y2 = 0.001, 0.059, 0.001, 0.24
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    ax1.indicate_inset_zoom(axins)

    plt.legend()
    plt.tight_layout()
    # TODO investigate why this causes second x axis to disappear
    # ut.enhance_plot(fig=fig, show=show, filename=filename, params_dict=params_dict, dir_name=directory)
    if show:
        fig.show()
    if filename is not None:
        fig.savefig(os.path.join(directory, filename))
    return fig
