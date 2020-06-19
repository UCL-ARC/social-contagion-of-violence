import src.common as co
import pytest
import numpy as np


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
    timestamps_nodes = np.array(['a', 'c', 'a', 'a'])
    node_list = ['c', 'b', 'a']
    expected = [np.array([2]), np.array([]), np.array([1, 1, 3])]
    actual = co.group_sort_timestamps(timestamps, timestamps_nodes, node_list)
    for i, j in zip(expected, actual):
        np.testing.assert_array_almost_equal(i, j)
