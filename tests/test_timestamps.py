import numpy as np
import networkx as nx
import src.timestamps as ts
import pytest

start_time = 2
t1 = [np.array([1, 2]), np.array([]), np.array([1])]


def test_get_infected_nodes():
    np.testing.assert_array_equal(ts.get_infected_nodes(t1, [0, 3]), np.array([[1, 0, 1]]))


def test_get_infected_nodes_count():
    np.testing.assert_array_equal(ts.get_infected_nodes(t1, [0, 3], count=True), np.array([[2, 0, 1]]))


def test_get_infected_nodes_range():
    np.testing.assert_array_equal(ts.get_infected_nodes(t1, range(3)), np.array([[0, 0, 0], [1, 0, 1]]))


def test_group_sort_timestamps():
    timestamps = np.array([1, 2, 3, 1])
    timestamps_nodes = np.array([0, 2, 0, 0])
    node_list = [2, 1, 0]
    expected = [np.array([2]), np.array([]), np.array([1, 1, 3])]
    actual = ts.group_sort_timestamps(timestamps, timestamps_nodes, node_list)
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)


def test_group_sort_timestamps_node_list_range():
    timestamps = np.array([1, 2, 3, 1])
    timestamps_nodes = np.array([0, 3, 0, 0])
    expected = [np.array([1, 1, 3]), np.array([]), np.array([]), np.array([2])]
    node_list = list(range(3))
    actual = ts.group_sort_timestamps(timestamps, timestamps_nodes, node_list)
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)


def test_get_clustering():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 2)])
    assert ts.calc_clustering(t1, g)[0] * 3, -1
    assert ts.calc_clustering(t1, g)[1], pytest.approx(-0.8181)
