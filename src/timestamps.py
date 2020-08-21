"""
Functions to manipulate timestamps
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import networkx as nx
import src.utilities as ut


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


def plot_timestamps(timestamps, network, simulation=None, node_list=None,
                    show=True, filename=None, params_dict=None, directory='results'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    counts, bins = np.histogram(np.concatenate(timestamps), bins=25)
    ax1.hist(bins[:-1], bins, weights=counts / bins[1])
    ax1.axhline(y=np.mean(counts / bins[1]), linewidth=2, color='black')

    if simulation is not None:
        lm, bs = mean_intensity(simulation) * simulation.n_nodes
        ax1.axhline(y=lm, linewidth=2, color='b', label=f'Est. mean events: {ut.round_to_n(lm, 3)}')
        ax1.axhline(y=bs, linewidth=2, color='r', label=f'Est. base events: {ut.round_to_n(bs, 3)}')
        ax1.legend(loc='lower left')
    ax1.set_title('Events per time unit over time')
    ax1.set_xlabel(f'Time')
    ax1.set_ylabel('Events per time unit')

    if node_list is not None:
        ts_count = [len(timestamps[n]) for n in node_list]
    else:
        ts_count = [len(t) for t in timestamps]
    ax2.hist(ts_count)
    ax2.axvline(x=np.mean(ts_count), linewidth=2, color='black',
                label=f'Mean events: {ut.round_to_n(np.mean(ts_count), 3)}')
    event_assortativities = event_assortativity(timestamps, network)
    ax2.plot([], [], ' ', label=f"Event assortativity")
    ax2.plot([], [], ' ', label=f" occurrence: {np.round(event_assortativities[0], 3)}")
    ax2.plot([], [], ' ', label=f" count: {np.round(event_assortativities[1], 3)}")
    ax2.legend()
    ax2.set_title('Histogram of events per node')
    ax2.set_xlabel(f'Events per node')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    ut.enhance_plot(fig, show, filename, params_dict, directory)
    return fig, event_assortativities


def event_assortativity(timestamps, network, start_time=0, end_time=None):
    if end_time is None:
        end_time = np.max(np.concatenate(timestamps)) + 1
    infections = get_infected_nodes(timestamps, times=[start_time, end_time]).tolist()[0]
    nx.set_node_attributes(network, dict(enumerate(infections)), 'infections_occurence')
    infections = get_infected_nodes(timestamps, times=[start_time, end_time], count=True).tolist()[0]
    nx.set_node_attributes(network, dict(enumerate(infections)), 'infections_count')
    return (nx.numeric_assortativity_coefficient(network, 'infections_occurence'),
            nx.numeric_assortativity_coefficient(network, 'infections_count'))


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

    plt.tight_layout()
    ut.enhance_plot(fig, show, filename, params_dict, directory)
    return fig


def mean_intensity(simulation):
    """Compute the mean baseline and intensity vector for time-dependent baselines
    """
    get_norm = np.vectorize(lambda kernel: kernel.get_norm())
    kernel_norms = get_norm(simulation.kernels)
    base_norms = get_norm(simulation.baseline)
    mean_intensity = np.mean(inv(np.eye(simulation.n_nodes) - kernel_norms).dot(base_norms) / simulation.end_time)
    mean_baseline = np.mean(inv(np.eye(simulation.n_nodes)).dot(base_norms) / simulation.end_time)
    return np.array([mean_intensity, mean_baseline])
