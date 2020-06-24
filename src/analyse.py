import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tick.hawkes import HawkesExpKern

import src.common as co


def plot_network(graph_or_adj, min_edges=3, show=True, filename=None, params_dict=None, directory='results'):
    g = co.to_graph(graph_or_adj)
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0:-1])
    ax2 = fig.add_subplot(gs[0, 2])

    # plot network
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=10, alpha=0.4, ax=ax1, )
    nx.draw_networkx_edges(g, pos, alpha=0.4, ax=ax1)
    labels = {n: n for n, d in g.degree if d > min_edges}
    nx.draw_networkx_labels(g, labels=labels, pos=pos, ax=ax1, font_size=10)
    ax1.set_title(f'Network Diagram')
    ax1.axis('off')

    # plot degree rank
    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
    ax2.plot(degree_sequence, marker='.')
    ax2.set_title('Degree rank plot')
    ax2.set_xlabel(f'rank')
    ax2.set_ylabel('degree')

    co.enhance_plot(fig, show, filename, params_dict, directory)
    return fig, pos


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
    co.plot_mean_median(ax2, timestamps_count)
    ax2.legend()
    ax2.set_title('Histogram of events per node')
    ax2.set_xlabel(f'Events per node')
    ax2.set_ylabel('Frequency')

    co.enhance_plot(fig, show, filename, params_dict, directory)
    return fig


def simple_fits(timestamps, decays, verbose=True):
    timestamps_single = np.concatenate(timestamps)
    timestamps_single.sort()
    timestamps_list = list()
    timestamps_list.append(timestamps_single)
    learners = list()
    for decay in decays:
        learner = HawkesExpKern(decay, verbose=verbose)
        learner.fit(timestamps_list)
        learners.append(learner)
    return learners


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

    co.enhance_plot(fig, show, filename, params_dict, directory)
    return fig
