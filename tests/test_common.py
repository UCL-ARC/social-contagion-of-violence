import src.common as co
import pytest


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
