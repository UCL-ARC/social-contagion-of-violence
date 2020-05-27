import networkx as nx
import matplotlib.pyplot as plt

import src.common as co


# g = nx.Graph()
# g.add_nodes_from([0,1 ,3,2, ], size=1)
# g.add_nodes_from([ 4], size=2)
# g.add_edges_from([(0,1),  (4,0), (0,2), (0,3), ])
# print(nx.numeric_assortativity_coefficient(g, 'size'))
# print(nx.attribute_mixing_dict(g, 'size'))
# print(nx.numeric_mixing_matrix(g,'size'))
#
# pos = nx.spring_layout(g)
# # nx.draw(g, pos=pos, ax=ax1, node_size=20, alpha=0.2)
# color_map = {1:'blue',2:'red'}
# nx.draw_networkx_nodes(g, pos, node_size=20, node_color=[color_map[k] for i, k in g.nodes.data('size')])
# nx.draw_networkx_edges(g, pos, alpha=0.4, )
# nx.draw_networkx_labels(g,  pos=pos,  font_size=20)
# plt.show()

def set_homophily(timestamps, g, node_list=None):
    node_attribute = dict()
    if node_list is None:
        node_list = range(len(timestamps))
    for node, node_timestamps in zip(node_list, timestamps):
        node_attribute[node] = 1 if len(node_timestamps) == 0 else 2
    nx.set_node_attributes(g, node_attribute, 'feature')


def plot_homophily_network(graph_or_adj, pos=None, show=True, filename=None, params_dict=None, directory='results'):
    g = co.to_graph(graph_or_adj)
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
    co.enhance_plot(fig, show, filename, params_dict, directory)
    return fig, pos


def repeat_set_homophily(multi_timestamps, g):
    homophily_dist = []
    homophily_mixing = []
    for timestamps in multi_timestamps:
        set_homophily(timestamps, g)
        homophily_dist.append(nx.numeric_assortativity_coefficient(g, 'feature'))
        homophily_mixing.append(nx.attribute_mixing_dict(g, 'feature'))
    return homophily_dist, homophily_mixing


def plot_homophily_variation(multi_timestamps, g, show=True, filename=None, params_dict=None, directory='results'):
    homophily_dist, homophily_mixing = repeat_set_homophily(multi_timestamps, g)
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))

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
    co.enhance_plot(fig, show, filename, params_dict, directory)
    return fig
