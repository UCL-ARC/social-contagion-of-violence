import pytest
import numpy as np

import src.utilities as ms


@pytest.mark.parametrize("n,expected", [(1, '10'), (2, '12'), (3, '12')], ids=str)
def test_round_12_to_n(n, expected):
    assert ms.round_to_n(12, n) == expected


@pytest.mark.parametrize("n,expected", [(1, '0.001'), (2, '0.001')], ids=str)
def test_round_001_to_n(n, expected):
    assert ms.round_to_n(0.001, n) == expected


@pytest.mark.parametrize("n,expected", [(1, '-10'), (2, '-12'), (3, '-12')], ids=str)
def test_round_negative12_to_n(n, expected):
    assert ms.round_to_n(-12, n) == expected


@pytest.mark.parametrize("n,expected", [(1, '0'), (2, '0')], ids=str)
def test_round_0_to_n(n, expected):
    assert ms.round_to_n(0, n) == expected


def test_norm():
    a = np.array([[0, 1, 2], [1, 2, 3]])
    np.testing.assert_array_equal([[0, 1, 2], [0.5, 1, 1.5]], ms.norm(a) * 3)


def test_norm_zero():
    a = np.array([[0, 0, 1], [0, 0, 0]])
    np.testing.assert_array_equal(a, ms.norm(a))
