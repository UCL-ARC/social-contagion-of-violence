import src.common as co
import pytest
import numpy as np
import networkx as nx


@pytest.mark.parametrize("n,expected", [(1, '10'), (2, '12'), (3, '12')], ids=str)
def test_round_12_to_n(n, expected):
    assert co.round_to_n(12, n) == expected


@pytest.mark.parametrize("n,expected", [(1, '0.001'), (2, '0.001')], ids=str)
def test_round_001_to_n(n, expected):
    assert co.round_to_n(0.001, n) == expected


@pytest.mark.parametrize("n,expected", [(1, '-10'), (2, '-12'), (3, '-12')], ids=str)
def test_round_negative12_to_n(n, expected):
    assert co.round_to_n(-12, n) == expected


@pytest.mark.parametrize("n,expected", [(1, '0'), (2, '0')], ids=str)
def test_round_0_to_n(n, expected):
    assert co.round_to_n(0, n) == expected


def test_group_sort_timestamps():
    timestamps = np.array([1, 2, 3, 1])
    timestamps_nodes = np.array([0, 2, 0, 0])
    node_list = [2, 1, 0]
    expected = [np.array([2]), np.array([]), np.array([1, 1, 3])]
    actual = co.group_sort_timestamps(timestamps, timestamps_nodes, node_list)
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)


def test_group_sort_timestamps_node_list_range():
    timestamps = np.array([1, 2, 3, 1])
    timestamps_nodes = np.array([0, 3, 0, 0])
    expected = [np.array([1,1,3]), np.array([]), np.array([]),np.array([2])]
    node_list = list(range(3))
    actual = co.group_sort_timestamps(timestamps, timestamps_nodes,node_list)
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)


def test_to_adj_self_excite():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2)])
    actual = co.to_adj(g, self_excite=True)
    expected = np.array([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)
