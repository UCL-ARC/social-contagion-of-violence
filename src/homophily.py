import networkx as nx
import matplotlib.pyplot as plt

import src.common as co


def set_homophily(timestamps, g, node_list=None, use_length=False):
    node_attribute = dict()
    if node_list is None:
        node_list = range(len(timestamps))
    for node, node_timestamps in zip(node_list, timestamps):
        if use_length:
            node_attribute[node] = len(node_timestamps)
        else:
            node_attribute[node] = 1 if len(node_timestamps) > 0 else 0
    nx.set_node_attributes(g, node_attribute, 'feature')


def plot_homophily_network(graph_or_adj, pos=None, show=True, filename=None, params_dict=None, directory='results'):
    g = co.to_graph(graph_or_adj)
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0:-1])
    ax2 = fig.add_subplot(gs[0, 2])

    if pos is None:
        pos = nx.spring_layout(g)
    node_color = [k for i, k in g.nodes.data('feature')]
    nx.draw_networkx_nodes(g, pos, node_size=20, alpha=0.4, node_color=node_color, ax=ax1)
    nx.draw_networkx_edges(g, pos, alpha=0.4, ax=ax1)
    ax1.set_title(f'Network Diagram')
    ax1.axis('off')

    ax2.hist(dict(g.degree).values())
    ax2.set_title('Degree Histogram')
    ax2.set_xlabel(f'Degrees')
    ax2.set_ylabel('Frequency')
    ax2.set_yscale('log')
    co.enhance_plot(fig, show, filename, params_dict, directory)
    return fig, pos


def multi_assortativity(multi_timestamps, g, use_length=False):
    numeric_assortativity = []
    g_copy = g.copy()
    for timestamps in multi_timestamps:
        set_homophily(timestamps, g_copy, use_length=use_length)
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
