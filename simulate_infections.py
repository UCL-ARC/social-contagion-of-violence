import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os


def dict_string(d):
    return str(d).replace("{", "").replace("}", "").replace("'", "").replace(":", "=")


def set_results_directory(directory=None):
    if directory is not None:
        if isinstance(directory, dict):
            directory_name = dict_string(directory)
        else:
            directory_name = directory
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
    return directory_name


def plot_common(fig, show=True, filename=None, params_dict=None, directory='results'):
    if params_dict is not None:
        params_string = dict_string(params_dict)
        fig.text(0.01, 0.01, f'Parameters: {params_string}', fontsize=10, wrap=True)
        fig.subplots_adjust(bottom=0.2)
    if filename is not None:
        fig.savefig(os.path.join(directory, filename))
    if show:
        fig.show()


def plot_graph(g, min_edges=3, show=True, filename=None, params_dict=None, directory='results'):
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0:-1])
    ax2 = fig.add_subplot(gs[0, 2])

    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_size=10, alpha=0.4, ax=ax1, )
    nx.draw_networkx_edges(g, pos, alpha=0.4, ax=ax1)
    labels = {n: n for n, d in g.degree if d > min_edges}
    nx.draw_networkx_labels(g, labels=labels, pos=pos, ax=ax1, font_size=10)
    ax1.set_title(f'Network Diagram')
    ax1.axis('off')

    ax2.hist(dict(g.degree).values())
    ax2.set_title('Degree Histogram')
    ax2.set_xlabel(f'Degrees')
    ax2.set_ylabel('Frequency')

    plot_common(fig, show, filename, params_dict, directory)
    return fig, pos


def estimate_event_counts(G, base, end_time, alpha, nodes, nn=3):
    ratio = alpha / (1 - alpha)
    adj = nx.to_numpy_array(G)
    nnx = [sum(sum(np.linalg.matrix_power(adj, i))) * np.power(ratio, i) for i in range(1, nn + 1)]
    infections = base * end_time * (nodes + sum(nnx))
    print(f'Estimated base infections = {round(base * end_time * nodes)}')
    print(f'Estimated infections = {round(infections)}')
    print(f'Estimated infections with self-infection = {round(infections * (1 + ratio))}')


def repeat_simulations(simulation, n_simulations):
    multi_timestamps = []
    for i in range(n_simulations):
        simulation.reset()
        simulation.simulate()
        multi_timestamps.append(simulation.timestamps)
    return multi_timestamps


def plot_simulations(multi_timestamps, show=True, filename=None, params_dict=None, directory='results'):
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

    plot_common(fig, show, filename, params_dict, directory)
    return fig


def set_homophily(timestamps, g):
    node_attribute = dict()
    for node, node_timestamps in enumerate(timestamps):
        node_attribute[node] = 1 if len(node_timestamps) == 0 else 2
    nx.set_node_attributes(g, node_attribute, 'feature')


def plot_homophily_network(g, pos=None, show=True, filename=None, params_dict=None, directory='results'):
    fig = plt.figure(figsize=(8, 5))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0:-1])
    ax2 = fig.add_subplot(gs[0, 2])

    if pos is None:
        pos = nx.spring_layout(g)
    color_map = {1: 'blue', 2: 'red'}
    node_color = [color_map[k] for i, k in g.nodes.data('feature')]
    nx.draw_networkx_nodes(g, pos, node_size=20, alpha=0.4, node_color=node_color, ax=ax1)
    nx.draw_networkx_edges(g, pos, alpha=0.4, ax=ax1)
    ax1.set_title(f'Network Diagram')
    ax1.axis('off')

    ax2.hist(dict(g.degree).values())
    ax2.set_title('Degree Histogram')
    ax2.set_xlabel(f'Degrees')
    ax2.set_ylabel('Frequency')
    ax2.set_yscale('log')
    plot_common(fig, show, filename, params_dict, directory)
    return fig, pos


def plot_homophily_variation(multi_timestamps, g, show=True, filename=None, params_dict=None, directory='results'):
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))

    homophily_dist = []
    homophily_mixing = []
    for timestamps in multi_timestamps:
        set_homophily(timestamps, g)
        homophily_dist.append(nx.numeric_assortativity_coefficient(g, 'feature'))
        homophily_mixing.append(nx.attribute_mixing_dict(g, 'feature'))

    ax[0, 0].hist(homophily_dist)
    ax[0, 0].set_title(f'Assortativity coefficient', fontsize=10)

    ax[0, 1].hist([i.get(1, {}).get(2, 0) for i in homophily_mixing])
    ax[0, 1].set_title(f'Edges between dissimilar nodes', fontsize=10)

    ax[1, 0].hist([i.get(1, {}).get(1, 0) for i in homophily_mixing])
    ax[1, 0].set_title(f'Edges between resistant nodes', fontsize=10)

    ax[1, 1].hist([i.get(2, {}).get(2, 0) for i in homophily_mixing])
    ax[1, 1].set_title(f'Edges between susceptible nodes', fontsize=10)

    fig.suptitle('Histogram of various assortativity measures', )
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plot_common(fig, show, filename, params_dict, directory)
    return fig
