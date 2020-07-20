"""
Functions to manipulate timestamps
"""

import matplotlib.pyplot as plt
import numpy as np

import src.utilities as ut


def group_sort_timestamps(timestamps, timestamps_nodes, node_list):
    return [np.sort(timestamps[timestamps_nodes == node]) for node in node_list]


def get_infected_nodes(timestamps, times, count=False):
    infected_nodes = np.zeros([len(times) - 1, len(timestamps)])
    for node, t in enumerate(timestamps):
        for i, time in enumerate(times[:-1]):
            values = np.size(t[np.logical_and(t >= time, t < times[i + 1])])
            if count:
                infected_nodes[i, node] = values
            else:
                if values > 0:
                    infected_nodes[i, node] = 1
    return infected_nodes


def plot_timestamps(timestamps, node_list=None, show=True, filename=None, params_dict=None, directory='results',
                    ax1_kw=None, ax2_kw=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    if ax1_kw is None: ax1_kw = {}
    if ax2_kw is None: ax2_kw = {}

    ax1.hist(np.concatenate(timestamps), **ax1_kw)
    ax1.set_title('Histogram of event timestamp')
    ax1.set_xlabel(f'Time')
    ax1.set_ylabel('Frequency')

    if node_list is not None:
        timestamps_count = [len(timestamps[n]) for n in node_list]
    else:
        timestamps_count = [len(t) for t in timestamps]
    ax2.hist(timestamps_count, **ax2_kw)
    ut.plot_mean_median(ax2, timestamps_count)
    ax2.legend()
    ax2.set_title('Histogram of events per node')
    ax2.set_xlabel(f'Events per node')
    ax2.set_ylabel('Frequency')

    ut.enhance_plot(fig, show, filename, params_dict, directory)
    return fig


def repeat_simulations(simulation, n_simulations):
    # NOTE: There is a multi-threaded solution but it's slower on my environment
    # multi = SimuHawkesMulti(contagion_simu, n_realizations, n_threads=0)
    # multi.simulate()
    # contagion_timestamps = multi.timestamps

    multi_timestamps = []
    for i in range(n_simulations):
        simulation.reset()
        simulation.simulate()
        multi_timestamps.append(simulation.timestamps)
        print("Total Jumps: ", simulation.n_total_jumps)
    return multi_timestamps


def plot_multi_timestamps(multi_timestamps, show=True, filename=None, params_dict=None, directory='results'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    ax1.hist([len(np.concatenate(timestamps)) for timestamps in multi_timestamps])
    ax1.set_title('Histogram of event count')
    ax1.set_xlabel(f'Event Counts')
    ax1.set_ylabel('Frequency')

    for timestamps in multi_timestamps:
        ax2.hist([len(node_timestamps) for node_timestamps in timestamps], alpha=0.3)
    ax2.set_title('Histogram of event count per node')
    ax2.set_xlabel(f'Event Counts per Node')
    ax2.set_ylabel('Frequency')

    ut.enhance_plot(fig, show, filename, params_dict, directory)
    return fig
