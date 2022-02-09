import numpy as np
import src.shuffle as sh
import pytest
import networkx as nx

data_diff_vectors = [
    ([1], [2], [1], [1]),
    (np.array([1]), np.array([2]), np.array([1]), np.array([1])),
    ([1], [1, 2], [0, 1], [0]),
    ([1, 4], [1, 2], [0, 1, 3, 2], [0]),
    ([1], [], [], []),
    ([], [1.5], [], []),
    ([], [], [], []),
]
st = sh.ShuffleTest(None)


@pytest.mark.parametrize("v1,v2,expected,ignore", data_diff_vectors, ids=str)
def test_diff_vectors(v1, v2, expected, ignore):
    np.testing.assert_array_equal(st._diff_vectors(v1, v2), expected)


@pytest.mark.parametrize("v1,v2,ignore,expected", data_diff_vectors, ids=str)
def test_diff_vectors_smallest(v1, v2, ignore, expected):
    np.testing.assert_array_equal(st._diff_vectors(v1, v2, smallest=True), expected)


data_diff_vector = [
    ([1, 4, 6], [3, 5, 2], [2]),
    ([1], [], []),
    ([], [], []),
    (np.array([1, 2]), np.array([1]), np.array([1])),
]


@pytest.mark.parametrize("v,expected,ignore", data_diff_vector, ids=str)
def test_diff_vector(v, expected, ignore):
    np.testing.assert_array_equal(st._diff_vector(v), expected)


@pytest.mark.parametrize("v,ignore,expected", data_diff_vector, ids=str)
def test_diff_vector_smallest(v, ignore, expected):
    np.testing.assert_array_equal(st._diff_vector(v, smallest=True), expected)


adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
g = nx.from_numpy_array(adj)  # represents a graph with two nodes where node 1 is connected to node 0 and 2
g_excited = nx.from_numpy_array(adj + np.eye(3))
t0 = [[], [2], []]  # represents one event occurring at t=1 at node 1
t1 = [[1], [2], []]  # represents two events occurring at t=1 at node 0 and t=2 at node 1
t2 = [[1], [], [2]]  # represents two events occurring at t=1 at node 0 and t=2 at node 2
t3 = [[1, 2], [], []]  # represents two events occurring at t=1 and t=2 at node 0
t4 = [np.array([]), np.array([2, 3, 4, 5]), np.array([1, ])]  # represents 4 events occurring at node1, 1 event at node2


@pytest.mark.parametrize("timestamps,expected", [(t0, []), (t1, [1]), (t2, []), (t3, [])], ids=str)
def test_diff_neighbours(timestamps, expected):
    st = sh.ShuffleTest(g)
    np.testing.assert_array_equal(st._diff_neighbours(timestamps), np.array(expected))


@pytest.mark.parametrize("timestamps,expected", [(t0, []), (t1, [1]), (t2, []), (t3, [])], ids=str)
def test_diff_neighbours_smallest(timestamps, expected):
    st = sh.ShuffleTest(g)
    np.testing.assert_array_equal(st._diff_neighbours(timestamps, smallest=True), np.array(expected))


@pytest.mark.parametrize("timestamps,expected", [(t0, []), (t1, [1]), (t2, []), (t3, [1])], ids=str)
def test_diff_neighbours_self_excitation(timestamps, expected):
    st = sh.ShuffleTest(g_excited, )
    np.testing.assert_array_equal(st._diff_neighbours(timestamps, ), np.array(expected))


@pytest.mark.parametrize("timestamps,expected", [(t0, []), (t1, [1]), (t2, []), (t3, [1])], ids=str)
def test_diff_neighbours_self_excitation_smallest(timestamps, expected):
    st = sh.ShuffleTest(g_excited, )
    np.testing.assert_array_equal(st._diff_neighbours(timestamps, smallest=True), np.array(expected))


def test_shuffle_timestamps_shuffle_nodes():
    st = sh.ShuffleTest(g, seed=1)
    actual = st.shuffle_timestamps(t4, shuffle_nodes=True, )
    expected = [np.array([3., 4.]), np.array([]), np.array([1., 2., 5.])]
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)


def test_shuffle_timestamps_keep_nodes():
    st = sh.ShuffleTest(g, seed=1)
    actual = st.shuffle_timestamps(t4, )
    expected = [np.array([]), np.array([1, 2, 3, 4]), np.array([5])]
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)


def test_shuffle_timestamps_node_list():
    st = sh.ShuffleTest(g, seed=1)
    actual = st.shuffle_timestamps(t4, shuffle_nodes=[1, 2], )
    expected = [np.array([]), np.array([3, 4]), np.array([1, 2, 5])]
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)


def test_repeat_shuffle_diff():
    st = sh.ShuffleTest(g, seed=10)
    actual = st._repeat_shuffle_diff(t4, shuffles=3, shuffle_nodes=True, )
    expected = [np.array([2., 1., 3., 4.]), np.array([4., 3., 2., 3., 1., 2.]), np.array([1., 3., 2., 1., 4., 1.])]
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)


def test_repeat_shuffle_diff_keep_nodes():
    st = sh.ShuffleTest(g, seed=1)
    actual = st._repeat_shuffle_diff(t4, shuffles=3, )
    expected = [np.array([4., 3., 2., 1.]), np.array([3., 2., 1., 1.]), np.array([1., 1., 2., 3.])]
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)


def test_repeat_shuffle_diff_node_list():
    st = sh.ShuffleTest(g, seed=1)
    actual = st._repeat_shuffle_diff(t4, shuffles=3, shuffle_nodes=[1, 2], )
    expected = [np.array([2., 1., 2., 3., 2., 1.]), np.array([3., 2., 1., 4., 3., 2.]), np.array([4., 3., 2., 1.])]
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)
