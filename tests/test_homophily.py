import numpy as np
import networkx as nx

import src.homophily as ho

timestamps = [np.array([]), np.array([1, 2]), np.array([1])]

def test_set_homophily():
    g = nx.Graph();
    g.add_edges_from([(0, 1), (0, 2)])
    ho.set_homophily(timestamps, g)
    assert [0, 1, 1] == [k for i, k in g.nodes.data('feature')]


def test_set_homophily_node_list():
    g = nx.Graph();
    g.add_edges_from([(0, 1), (0, 3)])
    ho.set_homophily(timestamps, g, node_list=(0, 1, 3))
    assert [0, 1, 1] == [k for i, k in g.nodes.data('feature')]


def test_set_homophily_use_length():
    g = nx.Graph();
    g.add_edges_from([(0, 1), (0, 2)])
    ho.set_homophily(timestamps, g, use_length=True)
    assert [0, 2, 1] == [k for i, k in g.nodes.data('feature')]
