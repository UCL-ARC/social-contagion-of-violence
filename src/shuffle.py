import numpy as np
from itertools import combinations, product
import matplotlib.pyplot as plt
import sys

import src.utilities as ut
import src.timestamps as ts


# TODO speed up performance

class ShuffleTest:
    """Takes a NetworkX network as an input, then infers the Hawkes parameters according to
    the event data provided in the fit method. It can also calculate the node kernel intensity
    at specified time intervals for a specific event data set, which can be used to predict the
    contagion risk.
    """

    def __init__(self, network, verbose=False, seed=None):
        self.network = network
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.diff = None
        self.zscore = None

    def _diff_vectors(self, v1, v2, smallest=False):
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

    def _diff_vector(self, v, smallest=False):
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

    def _diff_neighbours(self, timestamps, smallest=False):
        diffs = []
        for n1, n2 in self.network.edges:
            if n1 == n2:
                diffs.append(self._diff_vector(timestamps[n1], smallest))
            else:
                diffs.append(self._diff_vectors(timestamps[n1], timestamps[n2], smallest))
        if len(diffs) > 0:
            return np.concatenate(diffs)
        else:
            return np.array([])

    def shuffle_timestamps(self, timestamps, shuffle_nodes=False):
        nodes = len(timestamps)
        timestamps_copy = np.concatenate(timestamps)
        self.rng.shuffle(timestamps_copy)
        if shuffle_nodes is True:
            timestamps_nodes = self.rng.integers(nodes, size=len(timestamps_copy))
        elif isinstance(shuffle_nodes, list):
            timestamps_nodes = self.rng.choice(shuffle_nodes, len(timestamps_copy))
        elif shuffle_nodes is False:
            timestamps_nodes = np.concatenate([np.ones(len(j), dtype=int) * i for i, j in enumerate(timestamps)])
        else:
            raise TypeError("shuffle_nodes needs to be bool or list")
        t = ts.group_sort_timestamps(timestamps_copy, timestamps_nodes, range(nodes))
        return t

    def _repeat_shuffle_diff(self, timestamps, shuffles, smallest=False, shuffle_nodes=False, ):
        # timestamps is a list of np.arrays representing timestamps at a node
        # all nodes must be present in the list
        # a node with zero timestamps is represented by an empty np.array
        shuffled_diffs = []
        for repetition in range(shuffles):
            t = self.shuffle_timestamps(timestamps, shuffle_nodes=shuffle_nodes)
            shuffled_diffs.append(self._diff_neighbours(t, smallest))
            if self.verbose:
                if repetition % 100 == 0:
                    print(f'repetition: {repetition}')
        return shuffled_diffs

    def _plot_timestamp_differences(self, ax, ts_diffs1, ts_diffs2):
        ax.hist(ts_diffs1, alpha=0.4, label='Observed', color='black', bins=25)
        ax.hist(ts_diffs2, alpha=0.4, label='Shuffled', bins=25)
        ax.legend()
        ax.set_title('a) Min. time between adjacent events')
        ax.set_xlabel(f'Time difference')
        ax.set_ylabel('Frequency')

    def _plot_mean_timestamp_differences(self, ax, ts_diffs, shuffled_diffs):
        ax.axvline(x=np.average(ts_diffs), linewidth=2, color='black',
                   label=f'Observed: {ut.round_to_n(np.average(ts_diffs), 3)}')
        # Average difference between shuffled timestamps
        avg_shuffled_diffs = [np.nanmean(shuffled_diff) for shuffled_diff in shuffled_diffs if len(shuffled_diff) > 1]
        ax.hist(avg_shuffled_diffs, bins=25)
        ax.axvline(x=np.nanmean(avg_shuffled_diffs), linewidth=2, color='r',
                   label=f'Shuffled: {ut.round_to_n(np.average(avg_shuffled_diffs), 3)}')

        self.diff = np.nanmean(avg_shuffled_diffs) - np.average(ts_diffs)
        self.zscore = self.diff / np.nanstd(avg_shuffled_diffs)
        ax.plot([], [], ' ', label=f'Z-score: {ut.round_to_n(self.zscore, 2)}')

        ax.legend(framealpha=0)
        ax.set_title('b) Mean min. time between adjacent events')
        ax.set_xlabel(f'Mean time difference')
        ax.set_ylabel('Frequency')

    def plot_shuffle_test(self, timestamps, shuffles, smallest=True, shuffle_nodes=False,
                          show=True, filename=None, params_dict=None, directory='results'):
        ts_diffs = self._diff_neighbours(timestamps, smallest)
        shuffled_diffs = self._repeat_shuffle_diff(timestamps, shuffles, smallest, shuffle_nodes,)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        self._plot_timestamp_differences(ax1, ts_diffs, shuffled_diffs[0])
        self._plot_mean_timestamp_differences(ax2, ts_diffs, shuffled_diffs)
        plt.tight_layout()
        ut.enhance_plot(fig, show, filename, params_dict, directory)
        return fig


def shuffle_test(events, network, t_col, node_col, n_perm):
    events_ind = events.reset_index(drop=True)
    events_ind = events_ind.reset_index()
    t_arr = events_ind[t_col].values
    t_diffs = np.abs(t_arr[:, np.newaxis] - t_arr[np.newaxis, :])
    # node_event_ixs = events_ind.groupby(node_col).indices
    node_event_ixs = events_ind.groupby(node_col)['index'].apply(list).to_dict()
    all_pairs = []
    for n1, n2 in network.edges:
        # Check whether both nodes experienced incident
        if n1 in node_event_ixs and n2 in node_event_ixs:
            n1_ixs = node_event_ixs[n1]
            n2_ixs = node_event_ixs[n2]
            edge_pairs = list(product(n1_ixs, n2_ixs))
            all_pairs.append(edge_pairs)
    min_diffs_obs = []
    for edge_pairs in all_pairs:
        min_diff = np.min([t_diffs[i, j] for (i, j) in edge_pairs])
        min_diffs_obs.append(min_diff)
    min_diffs_obs_mean = np.mean(min_diffs_obs)
    min_diffs_obs_median = np.median(min_diffs_obs)
    # perm = np.arange(t_diffs.shape[0])
    means_perm = []
    medians_perm = []
    # TODO: Fix indices
    for perm_index in range(n_perm):
        # TODO Use updated rng
        # perm = np.random.permutation(np.arange(t_diffs.shape[0]))
        perm = np.random.permutation(events_ind['index'].values)
        min_diffs_perm = []
        for edge_pairs in all_pairs:
            min_diff = min([t_diffs[perm[i], perm[j]] for (i, j) in edge_pairs])
            min_diffs_perm.append(min_diff)
        means_perm.append(np.mean(min_diffs_perm))
        medians_perm.append(np.median(min_diffs_perm))
    return min_diffs_obs_mean, min_diffs_obs_median, means_perm, medians_perm


def shuffle_test_nocache(events, network, t_col, node_col, n_perm):
    events_ind = events.reset_index(drop=True)
    events_ind = events_ind.reset_index()
    t_arr = events_ind[t_col].values
    # node_event_ixs = events_ind.groupby(node_col).indices
    node_event_ixs = events_ind.groupby(node_col)['index'].apply(list).to_dict()
    all_pairs = []
    for n1, n2 in network.edges:
        # Check whether both nodes experienced incident
        if n1 in node_event_ixs and n2 in node_event_ixs:
            n1_ixs = node_event_ixs[n1]
            n2_ixs = node_event_ixs[n2]
            edge_pairs = [(i, j) for (i, j) in product(n1_ixs, n2_ixs) if i!=j]
            all_pairs.append(edge_pairs)
    min_diffs_obs = []
    for edge_pairs in all_pairs:
        min_diff = np.min(np.abs([(t_arr[i] - t_arr[j]) for (i, j) in edge_pairs]))
        min_diffs_obs.append(min_diff)
    min_diffs_obs_mean = np.mean(min_diffs_obs)
    min_diffs_obs_median = np.median(min_diffs_obs)
    # perm = np.arange(t_diffs.shape[0])
    means_perm = []
    medians_perm = []
    # TODO: Fix indices
    for perm_index in range(n_perm):
        # TODO Use updated rng
        # perm = np.random.permutation(np.arange(t_diffs.shape[0]))
        perm = np.random.permutation(events_ind['index'].values)
        min_diffs_perm = []
        for edge_pairs in all_pairs:
            min_diff = np.min(np.abs([(t_arr[perm[i]] - t_arr[perm[j]]) for (i, j) in edge_pairs]))
            min_diffs_perm.append(min_diff)
        means_perm.append(np.mean(min_diffs_perm))
        medians_perm.append(np.median(min_diffs_perm))
    return min_diffs_obs_mean, min_diffs_obs_median, means_perm, medians_perm