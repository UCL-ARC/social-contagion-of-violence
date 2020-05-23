import numpy as np
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

import simulate_infections as si


def diff_vectors(v1, v2):
    x = np.reshape(v1, (len(v1), 1))
    y = np.reshape(v2, (len(v2), 1)).transpose()
    z = np.absolute(y - x)
    if len(z) > 0:
        return np.concatenate(z)
    else:
        return np.array([])


def diff_vector(v):
    return np.array([np.abs(n1 - n2) for n1, n2 in combinations(v, 2)])


def diff_neighbours(graph_or_adj, timestamps, ):
    g = si._graph(graph_or_adj)
    diffs = []
    for n1, n2 in g.edges:
        if n1 == n2:
            diffs.append(diff_vector(timestamps[n1]))
        else:
            diffs.append(diff_vectors(timestamps[n1], timestamps[n2]))
    if len(diffs) > 0:
        return np.concatenate(diffs)
    else:
        return np.array([])


def group_timestamps_by_node(timestamps, timestamps_nodes, node_list):
    return [timestamps[np.where(timestamps_nodes == node)] for node in node_list]


def shuffle_timestamps(timestamps, seed=None):
    nodes = len(timestamps)
    timestamps_copy = np.concatenate(timestamps)
    rng = np.random.default_rng(seed)
    rng.shuffle(timestamps_copy)
    timestamps_nodes = rng.integers(nodes, size=len(timestamps_copy))
    t = group_timestamps_by_node(timestamps_copy, timestamps_nodes, range(nodes))
    return t


def repeat_shuffle_diff(adj, timestamps, repetitions, seed=None, verbose=False):
    # timestamps is a list of np.arrays representing timestamps at a node
    # all nodes must be present in the list
    # a node with zero timestamps is represented by an empty np.array
    shuffled_diffs = []
    for repetition in range(repetitions):
        t = shuffle_timestamps(timestamps, seed=seed)
        shuffled_diffs.append(diff_neighbours(adj, t))
        if seed is not None:
            seed = seed + 1
        if verbose:
            if repetition % 100 == 0:
                print(f'repetition: {repetition}')
    return shuffled_diffs


def plot_diff_comparison(timestamps, shuffled_diffs, show=True, filename=None, params_dict=None, directory='results'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    ax1.hist(timestamps, alpha=0.5)
    ax1.hist(shuffled_diffs[0], alpha=0.5)
    ax1.set_title('Histogram of timestamp differences')
    ax1.set_xlabel(f'Timestamp difference')
    ax1.set_ylabel('Frequency')

    ax2.hist([np.average(shuffled_diff) for shuffled_diff in shuffled_diffs])
    ax2.axvline(x=np.average(timestamps), linewidth=2, color='r')
    ax2.set_title('Histogram of average timestamp differences')
    ax2.set_xlabel(f'Timestamp difference')
    ax2.set_ylabel('Frequency')

    si.plot_common(fig, show, filename, params_dict, directory)
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
