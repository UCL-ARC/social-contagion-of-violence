import numpy as np
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

import src.common as co


def diff_vectors(v1, v2, smallest=False):
    # Note: Assumes that number of timestamps per node is small so no need to use faster algorithm
    x = np.reshape(v1, (len(v1), 1))
    y = np.reshape(v2, (len(v2), 1)).transpose()
    z = np.absolute(y - x)
    if z.size > 0:
        if smallest:
            return np.array([np.min(z)])
        return np.concatenate(z)
    return np.array([])


def diff_vector(v, smallest=False):
    # Note: Assumes that number of timestamps per node is small so no need to use faster algorithm
    z = np.array([np.abs(n1 - n2) for n1, n2 in combinations(v, 2)])
    if smallest and len(z) > 0:
        return np.array([np.min(z)])
    return z


def diff_neighbours(graph_or_adj, timestamps, smallest=False):
    g = co.to_graph(graph_or_adj)
    diffs = []
    for n1, n2 in g.edges:
        if n1 == n2:
            diffs.append(diff_vector(timestamps[n1], smallest))
        else:
            diffs.append(diff_vectors(timestamps[n1], timestamps[n2], smallest))
    if len(diffs) > 0:
        return np.concatenate(diffs)
    else:
        return np.array([])


def shuffle_timestamps(timestamps, shuffle_nodes=False, seed=None):
    nodes = len(timestamps)
    timestamps_copy = np.concatenate(timestamps)
    rng = np.random.default_rng(seed)
    rng.shuffle(timestamps_copy)
    if shuffle_nodes:
        timestamps_nodes = rng.integers(nodes, size=len(timestamps_copy))
    else:
        timestamps_nodes = np.concatenate([np.ones(len(j), dtype=int) * i for i, j in enumerate(timestamps)])
    t = co.group_sort_timestamps(timestamps_copy, timestamps_nodes, range(nodes))
    return t


def repeat_shuffle_diff(adj, timestamps, shuffles, smallest=False, shuffle_nodes=False, seed=None, verbose=False):
    # timestamps is a list of np.arrays representing timestamps at a node
    # all nodes must be present in the list
    # a node with zero timestamps is represented by an empty np.array
    shuffled_diffs = []
    for repetition in range(shuffles):
        t = shuffle_timestamps(timestamps=timestamps, shuffle_nodes=shuffle_nodes, seed=seed)
        shuffled_diffs.append(diff_neighbours(adj, t, smallest))
        if seed is not None:
            seed = seed + 1
        if verbose:
            if repetition % 100 == 0:
                print(f'repetition: {repetition}')
    return shuffled_diffs


def plot_timestamp_differences(ax, ts_diffs1, ts_diffs2):
    ax.hist(ts_diffs1, alpha=0.4, label='actual')
    ax.hist(ts_diffs2, alpha=0.4, label='shuffled')
    ax.legend()
    ax.set_title('Histogram of timestamp differences')
    ax.set_xlabel(f'Timestamp difference')
    ax.set_ylabel('Frequency')


def plot_average_timestamp_differences(ax, ts_diffs, shuffled_diffs):
    # Average difference between shuffled timestamps
    avg_shuffled_diffs = [np.nanmean(shuffled_diff) for shuffled_diff in shuffled_diffs if len(shuffled_diff) > 1]
    ax.hist(avg_shuffled_diffs, label='shuffled')
    co.plot_mean_median(ax, avg_shuffled_diffs)
    ax.axvline(x=np.average(ts_diffs), linewidth=2, color='black',
               label=f'actual: {co.round_to_n(np.average(ts_diffs), 3)}')

    percentage_diff = np.abs(np.average(ts_diffs) - np.nanmean(avg_shuffled_diffs)) * 100 / \
                      np.nanmean(avg_shuffled_diffs)
    ax.plot([], [], ' ', label=f'%diff: {co.round_to_n(percentage_diff, 3)}')
    ax.plot([], [], ' ', label=f'stdev: {co.round_to_n(np.nanstd(avg_shuffled_diffs), 3)}')

    ax.legend()
    ax.set_title('Histogram of average shuffled timestamp differences')
    ax.set_xlabel(f'Timestamp difference')
    ax.set_ylabel('Frequency')


def plot_shuffle_test(adj, timestamps, shuffles, smallest=False, shuffle_nodes=False, seed=None, verbose=False,
                      show=True, filename=None, params_dict=None, directory='results'):
    ts_diffs = diff_neighbours(adj, timestamps, smallest)
    shuffled_diffs = repeat_shuffle_diff(adj, timestamps, shuffles, smallest, shuffle_nodes, seed, verbose)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    plot_timestamp_differences(ax1, ts_diffs, shuffled_diffs[0])
    plot_average_timestamp_differences(ax2, ts_diffs, shuffled_diffs)
    co.enhance_plot(fig, show, filename, params_dict, directory)
    return fig


def diff_nodes(timestamps, g, threshold):
    infected_nodes = np.where(np.array([len(t) for t in timestamps]) > 0)[0]
    infected_nodes_comb = combinations(infected_nodes, 2)
    lengths = []
    for n in infected_nodes_comb:
        min_diff = np.min(diff_vectors(timestamps[n[0]], timestamps[n[1]]))
        if min_diff < threshold:
            lengths.append(nx.shortest_path_length(g, n[0], n[1]))
    return lengths
