import numpy as np
import networkx as nx
import os
from math import log10, floor


def dict_string(d):
    return str(d).replace("{", "").replace("}", "").replace("'", "").replace(":", "=")


def set_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def enhance_plot(fig, show=True, filename=None, params_dict=None, dir_name='analysis/results'):
    if params_dict is not None:
        params_string = dict_string(params_dict)
        fig.text(0.01, 0.01, f'Parameters: {params_string}', fontsize=10, wrap=True)
        fig.subplots_adjust(bottom=0.2)
    if filename is not None:
        fig.savefig(os.path.join(dir_name, filename))
    if show:
        fig.show()


def to_graph(graph_or_adj):
    if isinstance(graph_or_adj, nx.Graph):
        return graph_or_adj
    elif isinstance(graph_or_adj, np.ndarray):
        return nx.from_numpy_array(graph_or_adj)
    elif isinstance(graph_or_adj, list):
        return nx.from_numpy_array(np.array(graph_or_adj))
    else:
        raise TypeError("graph_or_adj needs to be networkx graph, numpy array or list")


def to_adj(graph_or_adj, self_excite=False):
    if isinstance(graph_or_adj, nx.Graph):
        adj = nx.to_numpy_array(graph_or_adj)
    elif isinstance(graph_or_adj, np.ndarray):
        adj = graph_or_adj
    elif isinstance(graph_or_adj, list):
        adj = np.array(graph_or_adj)
    else:
        raise TypeError("graph_or_adj needs to be networkx graph, numpy array or list")
    if self_excite:
        adj[np.where(np.eye(len(adj)))] = 1
    return adj


# round_to_n = lambda x, n: 0 if x==0 else round(x, -int(floor(log10(abs(x)))) + (n - 1))
def round_to_n(value, n):
    return str('{:g}'.format(float('{:.{p}g}'.format(value, p=n))))


def plot_mean_median(ax, data):
    ax.axvline(x=np.nanmean(data), linewidth=2, color='r', label=f'mean: {round_to_n(np.average(data), 3)}')
    ax.axvline(x=np.nanmedian(data), linewidth=2, color='g', label=f'median: {round_to_n(np.median(data), 3)}')


def group_sort_timestamps(timestamps, timestamps_nodes, node_list):
    return [np.sort(timestamps[np.where(timestamps_nodes == node)]) for node in node_list]
