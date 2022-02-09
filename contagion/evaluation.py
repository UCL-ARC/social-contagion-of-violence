import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import contagion.utilities as ut


def _top_percent(arr, x=1):
    """Flag the top x percent highest-risk nodes.

    If two nodes have the same risk, select the first node.
    """
    highest_risk = np.zeros(len(arr))
    n = int(np.ceil(len(arr) * x / 100))
    highest_risk[np.argpartition(arr, -n)[-n:]] = 1
    return highest_risk


def cm_sum_percent(y_true, risk, percentages, labels=(0, 1)):
    """Evaluate the predicted risk of models where risk is time-dependent, as a function of coverage percentage.

     The infected nodes and predicted risk must be calculated for the same time intervals and have the same array shape.
     The coverage percentage represents the percentage of the highest-risk nodes that are selected and converts the
     model risk into labels. The confusion matrix is calculated for every time-slice provided.
     The individual confusion matrices are then summed into a total confusion matrix.
     This is repeated for every specified coverage percentage value.
    """
    if y_true.shape != risk.shape:
        raise ValueError('y_true and risk should have the same shape')
    cm_time = np.zeros([len(percentages), 4], dtype=int)
    for i, percent in enumerate(percentages):
        for y_true_time, risk_time in zip(y_true, risk):
            y_pred_time = _top_percent(risk_time, percent)
            cm_time[i] += confusion_matrix(y_true_time, y_pred_time, labels=labels).flatten()
    return cm_time


def _plot_main(cms, percentages, time, ax):
    #  create secondary axes
    axy = ax.twinx()
    axx = axy.twiny()
    for data_name, data_values in cms.items():
        ax.plot(percentages / 100, data_values[:, 3] / np.sum(data_values[:, 2:], 1), '-o', label=data_name)
        axx.scatter(np.sum(data_values[:, [1, 3]], 1) / time, data_values[:, 3], label=data_name, )
    ax.set_xlabel('Coverage (proportion of highest-risk nodes) per time unit')
    ax.set_ylabel('Hit-rate over prediction time')
    # format secondary x-axis
    axx.set_xlabel('Highest-risk nodes selected per time unit')
    axx.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    axx.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    axx.spines['bottom'].set_position(('outward', 36))
    # format secondary y-axis
    axy.set_ylabel('Nodes correctly identified over prediction time')


def _plot_inset(cms, percentages, ax):
    axins = ax.inset_axes([0.6, 0.07, 0.37, 0.47])
    for data_name, data_values in cms.items():
        axins.plot(percentages / 100, data_values[:, 3] / np.sum(data_values[:, 2:], 1), '-o')
    # limits of axes
    x1, x2, y1, y2 = 0.001, 0.059, 0.001, 0.24
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    ax.indicate_inset_zoom(axins)


def plot_hit_rate(cms, percentages, time, show=True, filename=None, directory='results'):
    """Calculate and plot the hit rate as a function of coverage percentage from the confusion matrice(s).

    The coverage percentage values of [0.5, 1, 2, 5, 10, 20, 40, 60, 80, 100] were found to work the best.
    The hit rate (also known as recall) is the fraction of all infected nodes that were identified as highest-risk.
    The hit rate is a common metric for evaluating models where returning positive predictions is more important than
    potentially misclassifying predictions.

    :param cms:
    :param percentages:
    :param time:
    :param show:
    :param filename:
    :param directory:
    :return: the figure object
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    _plot_main(cms, percentages, time, ax=ax1)
    _plot_inset(cms, percentages, ax=ax1)
    plt.legend(loc='upper left', framealpha=0)
    plt.tight_layout()
    ut.enhance_plot(fig=fig, show=show, filename=filename, dir_name=directory)
    return fig
