import numpy as np
import networkx as nx

import src.homophily as ho


def test_set_homophily():
    g = nx.Graph();
    g.add_edges_from([(0, 1), (0, 2)])
    timestamps = [np.array([]), np.array([1, 2]), np.array([1])]
    ho.set_homophily(timestamps, g)
    assert [1, 2, 2] == [k for i, k in g.nodes.data('feature')]


def test_set_homophily_node_list():
    g = nx.Graph();
    g.add_edges_from([(0, 1), (0, 3)])
    timestamps = [np.array([]), np.array([1, 2]), np.array([1])]
    ho.set_homophily(timestamps, g, node_list=(0, 1, 3))
    assert [1, 2, 2] == [k for i, k in g.nodes.data('feature')]
