import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import src.common as co
from tick.base import TimeFunction
from tick.plot import plot_timefunction


def set_feature_timestamps(g, timestamps, use_length=False):
    for n in g.nodes:
        if use_length:
            g.nodes[n]['feature'] = len(timestamps[n])
        else:
            g.nodes[n]['feature'] = 1 if len(timestamps[n]) > 0 else 0
    return {i:k for i,k in g.nodes.data('feature')}


def set_homophily_random(g, values=None, node_centers=None, max_nodes=None, base=-1, seed=None):
    node_count = 0
    node_attribute = {i: base for i in g.nodes}
    if max_nodes is None:
        max_nodes = g.number_of_nodes()
    rng = np.random.default_rng(seed)
    candidate_nodes = rng.choice(list(g.nodes), max_nodes, replace=False)
    if node_centers is None:
        node_centers = candidate_nodes

    for node in node_centers:
        if values is None:
            value = node
        else:
            value = rng.choice(values)
        if node_count > max_nodes:
            break
        node_attribute[node] = value
        node_count += 1
        for n in g.neighbors(node):
            if n in candidate_nodes:
                if node_count > max_nodes:
                    break
                node_attribute[n] = value
                node_count += 1

    nx.set_node_attributes(g, node_attribute, 'feature')
    return node_attribute


def _peak_time_function(start, duration, intensity):
    t_values = np.array([start, start + duration])
    y_values = np.array([1 / duration, 0]) * intensity
    return TimeFunction((t_values, y_values), inter_mode=TimeFunction.InterConstRight)


def peak_time_functions(g, runtime, duration, intensity=1, base=0):
    tfs = list(range(g.number_of_nodes()))
    for n, value in g.nodes.data('feature'):
        if value > -1:
            tfs[n] = _peak_time_function(value * runtime / g.number_of_nodes(), duration, intensity)
        else:
            tfs[n] = base/runtime
    return tfs


def plot_time_functions(tfs, show=True, filename=None, params_dict=None, directory='results'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if not isinstance(tfs, list):
        tfs = [tfs]
    for tf in tfs:
        if isinstance(tf, TimeFunction):
            plot_timefunction(tf, ax=ax, show=False)
            if len(tfs) > 3:
                ax.get_legend().remove()
    ax.set_title(f'Time functions)', fontsize=10)
    co.enhance_plot(fig, show, filename, params_dict, directory)


def norm_time_functions(g, runtime, intensity):
    tfs = np.array([k for i, k in g.nodes.data('feature')])
    tfs = tfs * intensity * g.number_of_nodes() / (sum(tfs) * runtime)
    return tfs


def multi_assortativity(multi_timestamps, g, use_length=False):
    numeric_assortativity = []
    g_copy = g.copy()
    for timestamps in multi_timestamps:
        set_feature_timestamps(g_copy, timestamps, use_length=use_length)
        numeric_assortativity.append(nx.numeric_assortativity_coefficient(g_copy, 'feature'))
    return numeric_assortativity


def plot_homophily_variation(multi_timestamps, g, show=True, filename=None, params_dict=None, directory='results'):
    numeric_assortativity = multi_assortativity(multi_timestamps, g)
    numeric_assortativity_use_length = multi_assortativity(multi_timestamps, g, use_length=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    ax1.hist(numeric_assortativity)
    ax1.set_title(f'Numeric Assortativity coefficient (event occurence)', fontsize=10)

    ax2.hist(numeric_assortativity_use_length)
    ax2.set_title(f'Numeric Assortativity coefficient (event count)', fontsize=10)

    co.enhance_plot(fig, show, filename, params_dict, directory)
    return fig


def plot_homophily_network(graph_or_adj, min_edges=3, pos=None, show=True, filename=None, params_dict=None,
                           directory='results'):
    g = co.to_graph(graph_or_adj)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if pos is None:
        pos = nx.spring_layout(g)
    node_color = [k for i, k in g.nodes.data('feature')]
    nc = nx.draw_networkx_nodes(g, pos, node_size=20, alpha=0.4, node_color=node_color, ax=ax, cmap=plt.cm.jet)
    nx.draw_networkx_edges(g, pos, alpha=0.4, ax=ax)
    labels = {n: n for n, d in g.degree if d > min_edges}
    nx.draw_networkx_labels(g, labels=labels, pos=pos, ax=ax, font_size=12)
    ax.set_title(f'Network Diagram with varying feature')
    ax.axis('off')
    ax.plot([], [], ' ', label=f"Assortavity: {np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)}")
    ax.legend()
    plt.colorbar(nc)
    co.enhance_plot(fig, show, filename, params_dict, directory)
    return fig, pos
