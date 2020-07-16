import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import sys

import src.utilities as ut
import src.timestamps as ts


def diff_vectors(v1, v2, smallest=False):
    if len(v1) == 0 or len(v2) == 0:
        return np.array([])
    if smallest:
        a, b = 0, 0
        result = sys.maxsize
        while a < len(v1) and b < len(v2):
            diff = abs(v1[a] - v2[b])
            if diff < result:
                result = diff
            if v1[a] < v2[b]:
                a += 1
            else:
                b += 1
        return np.array([result])
    else:
        x = np.reshape(v1, (len(v1), 1))
        y = np.reshape(v2, (len(v2), 1)).transpose()
        z = np.absolute(y - x)
        return np.concatenate(z)


def diff_vector(v, smallest=False):
    if len(v) > 1:
        if smallest:
            # Find the min diff by comparing difference of all possible pairs in sorted array
            diff = sys.maxsize
            for i in range(len(v) - 1):
                if v[i + 1] - v[i] < diff:
                    diff = v[i + 1] - v[i]
            return np.array([diff])
        else:
            # Note: very slow for nodes with large number of timestamps > 100
            return np.array([np.abs(n1 - n2) for n1, n2 in combinations(v, 2)])
    return np.array([])


def diff_neighbours(g, timestamps, smallest=False):
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
    if shuffle_nodes is True:
        timestamps_nodes = rng.integers(nodes, size=len(timestamps_copy))
    elif isinstance(shuffle_nodes, list):
        timestamps_nodes = rng.choice(shuffle_nodes, len(timestamps_copy))
    elif shuffle_nodes is False:
        timestamps_nodes = np.concatenate([np.ones(len(j), dtype=int) * i for i, j in enumerate(timestamps)])
    else:
        raise TypeError("shuffle_nodes needs to be bool or list")
    t = ts.group_sort_timestamps(timestamps_copy, timestamps_nodes, range(nodes))
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
    ut.plot_mean_median(ax, avg_shuffled_diffs)
    ax.axvline(x=np.average(ts_diffs), linewidth=2, color='black',
               label=f'actual: {ut.round_to_n(np.average(ts_diffs), 3)}')

    percentage_diff = np.abs(np.average(ts_diffs) - np.nanmean(avg_shuffled_diffs)) * 100 / \
                      np.nanmean(avg_shuffled_diffs)
    ax.plot([], [], ' ', label=f'%diff: {ut.round_to_n(percentage_diff, 3)}')
    ax.plot([], [], ' ', label=f'stdev: {ut.round_to_n(np.nanstd(avg_shuffled_diffs), 3)}')

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
    ut.enhance_plot(fig, show, filename, params_dict, directory)
    return fig
