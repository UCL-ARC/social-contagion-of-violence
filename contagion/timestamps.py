"""
Functions to manipulate timestamps
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import networkx as nx
import contagion.utilities as ut


def group_sort_timestamps(timestamps, timestamps_nodes, node_list):
    return [np.sort(timestamps[timestamps_nodes == node]) for node in node_list]


def get_infected_nodes(timestamps, times, count=False):
    infected_nodes = np.zeros([len(times) - 1, len(timestamps)], dtype=int)
    for node, t in enumerate(timestamps):
        for i, time in enumerate(times[:-1]):
            values = np.size(t[np.logical_and(t >= time, t < times[i + 1])])
            if count:
                infected_nodes[i, node] = values
            else:
                if values > 0:
                    infected_nodes[i, node] = 1
    return infected_nodes


def timestamps_assortativity(timestamps, network, start_time=0, end_time=None):
    if end_time is None:
        end_time = np.max(np.concatenate(timestamps)) + 1
    infections = get_infected_nodes(timestamps, times=[start_time, end_time]).tolist()[0]
    nx.set_node_attributes(network, dict(enumerate(infections)), 'infections_occurence')
    infections = get_infected_nodes(timestamps, times=[start_time, end_time], count=True).tolist()[0]
    nx.set_node_attributes(network, dict(enumerate(infections)), 'infections_count')
    return (nx.numeric_assortativity_coefficient(network, 'infections_occurence'),
            nx.numeric_assortativity_coefficient(network, 'infections_count'))


def mean_intensity(simulation):
    """Compute the mean baseline and intensity vector for time-dependent baselines
    """
    get_norm = np.vectorize(lambda kernel: kernel.get_norm())
    kernel_norms = get_norm(simulation.kernels)
    base_norms = get_norm(simulation.baseline)
    mean_intensity = np.mean(inv(np.eye(simulation.n_nodes) - kernel_norms).dot(base_norms) / simulation.end_time)
    mean_baseline = np.mean(inv(np.eye(simulation.n_nodes)).dot(base_norms) / simulation.end_time)
    return np.array([mean_intensity, mean_baseline])


def plot_timestamps(timestamps, ts_assortativities, simulation=None, node_list=None,
                    show=True, filename=None, params_dict=None, directory='results'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    counts, bins = np.histogram(np.concatenate(timestamps), bins=25)
    ax1.hist(bins[:-1], bins, weights=counts / bins[1])
    ax1.axhline(y=np.mean(counts / bins[1]), linewidth=2, color='black')

    if simulation is not None:
        lm, bs = mean_intensity(simulation) * simulation.n_nodes
        ax1.axhline(y=lm, linewidth=2, color='b', label=f'Exp. events: {ut.round_to_n(lm, 3)}')
        ax1.axhline(y=bs, linewidth=2, color='r', label=f'Exp. base events: {ut.round_to_n(bs, 3)}')
        ax1.legend(loc='lower left')
    ax1.set_title('a) Events per unit time over time')
    ax1.set_xlabel(f'Time')
    ax1.set_ylabel('Events per unit time')

    if node_list is not None:
        ts_count = [len(timestamps[n]) for n in node_list]
    else:
        ts_count = [len(t) for t in timestamps]
    ax2.hist(ts_count)
    ax2.axvline(x=np.mean(ts_count), linewidth=2, color='black',
                label=f'Mean events: {ut.round_to_n(np.mean(ts_count), 3)}')
    ax2.plot([], [], ' ', label=f"Event assortativity")
    ax2.plot([], [], ' ', label=f" occurrence: {np.round(ts_assortativities[0], 3)}")
    ax2.plot([], [], ' ', label=f" count: {np.round(ts_assortativities[1], 3)}")
    ax2.legend()
    ax2.set_title('b) Histogram of events per node')
    ax2.set_xlabel(f'Events per node')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    ut.enhance_plot(fig, show, filename, params_dict, directory)
    return fig
