import numpy as np
import networkx as nx

import src.homophily as ho

timestamps = [np.array([1, 2]), np.array([]), np.array([1])]


def test_set_homophily_timestamps():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 2)])
    feature = ho.set_feature_timestamps(g, timestamps)
    assert feature == {0: 1, 1: 0, 2: 1}


def test_set_homophily_timestamps_use_length():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 2)])
    feature = ho.set_feature_timestamps(g, timestamps, use_length=True)
    assert feature == {0: 2, 1: 0, 2: 1}


def test_peak_time_function():
    tf = ho._peak_time_function(start=0.2, duration=1, intensity=0.5)
    assert tf.get_norm() == 0.5
    assert tf.value(0) == 0
    assert tf.value(1) == 0.5
    assert tf.value(2) == 0


def test_set_homophily_random():
    g = nx.Graph()
    g.add_edges_from([(1, 0), (1, 2), (3, 4)])
    feature = ho.set_homophily_random(g, node_centers=[1], )
    assert feature == {1: 1, 0: 1, 2: 1, 3: -1, 4: -1}


def test_set_homophily_random_max():
    g = nx.Graph()
    g.add_nodes_from(range(10))
    feature = ho.set_homophily_random(g, max_nodes=5, seed=1)
    assert feature == {0: 0, 1: -1, 2: 2, 3: 3, 4: -1, 5: -1, 6: 6, 7: -1, 8: 8, 9: -1}


def test_set_homophily_random_max_neighbors():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
    feature = ho.set_homophily_random(g, max_nodes=3, node_centers=[0], seed=3)
    assert feature == {0: 0, 1: -1, 2: 0, 3: -1, 4: 0}


def test_set_homophily_random_values():
    g = nx.Graph()
    g.add_edges_from([(1, 0), (1, 2), (3, 4)])
    feature = ho.set_homophily_random(g, node_centers=[1], values=[10], base=0)
    assert feature == {1: 10, 0: 10, 2: 10, 3: 0, 4: 0}


def test_peak_time_functions():
    # In this example only nodes 0,1,2 are prone to infection and they have the same time function profile because
    # they are neighbours of node 1 which peaks between 2 and 3 time units
    g = nx.Graph()
    g.add_edges_from([(1, 0), (1, 2), (3, 4)])
    nx.set_node_attributes(g, {1: 1, 0: 1, 2: 1, 3: -1, 4: -1}, 'feature')

    tfs = ho.peak_time_functions(g, runtime=10, duration=1, )
    np.testing.assert_array_equal(tfs[0].original_t, np.array([2, 3]))
    np.testing.assert_array_equal(tfs[1].original_t, np.array([2, 3]))
    np.testing.assert_array_equal(tfs[2].original_t, np.array([2, 3]))
    assert tfs[3] == 0
    assert tfs[4] == 0


def test_norm_time_functions():
    g = nx.Graph()
    g.add_edges_from([(1, 0), (1, 2), (3, 4)])
    nx.set_node_attributes(g, {1: 1, 0: 1, 2: 0, 3: 0, 4: 0}, 'feature')

    tfs = ho.norm_time_functions(g, runtime=5, intensity=2)
    np.testing.assert_array_equal(tfs, np.array([1, 1, 0, 0, 0]))
