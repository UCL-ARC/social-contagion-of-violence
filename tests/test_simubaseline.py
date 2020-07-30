import numpy as np
import pytest
from src import SimuBaseline


def test_SimuBaseline():
    bs = SimuBaseline(n_nodes=4, network_type='path', )
    assert list(bs.network.edges) == [(0, 1), (1, 2), (2, 3)]


def test_homophily():
    bs = SimuBaseline(n_nodes=4, network_type='path', )
    feature = bs._homophilic_feature(values=[1], node_centers=[0], )
    np.testing.assert_array_equal(feature, np.array([1, 1, 0, 0]))


def test_homophily_max():
    bs = SimuBaseline(n_nodes=4, network_type='star', )
    feature = bs._homophilic_feature(values=[1], node_centers=[0], max_nodes=2)
    np.testing.assert_array_equal(feature, np.array([1, 1, 0, 0, 0]))

    bs = SimuBaseline(n_nodes=6, network_type='path', seed=0)
    feature = bs._homophilic_feature(values=[1], node_centers=[1, 4], max_nodes=5)
    np.testing.assert_array_equal(feature, np.array([1, 1, 1, 1, 1, 0]))


def test_features_single():
    bs = SimuBaseline(n_nodes=5, network_type='path', seed=0)
    bs.simulate(proportions=[0.5], mean_mu=0.5, variation=1)
    np.testing.assert_array_equal(bs.features.T, [[1., 0., 0., 0.,1]])
    assert 0.5 == pytest.approx(np.average(bs.node_mu), rel=1e-2)


def test_features_multi():
    bs = SimuBaseline(n_nodes=3, network_type='path', seed=0)
    bs.simulate(proportions=[0.5, 0.5], mean_mu=0.5, variation=0)
    np.testing.assert_array_equal(bs.features.T, [[1., 0, 1.], [0., 0., 1.]])
    assert 0.5 == pytest.approx(np.average(bs.node_mu), rel=1e-2)
    np.testing.assert_array_almost_equal(bs.assortativity[:-1] * 3, [-3, -1])


def test_features_homophilic():
    bs = SimuBaseline(n_nodes=3, network_type='path', seed=0)
    bs.simulate(proportions=[0.5, 0.5], mean_mu=0.5, variation=1, homophilic=True)
    np.testing.assert_array_equal(bs.features.T, [[0., 0., 1.], [0., 1., 0.]])
    assert 0.5 == pytest.approx(np.average(bs.node_mu), rel=1e-2)
    np.testing.assert_array_almost_equal(bs.assortativity[:-1] * 3, [-1, -3])

# TODO Unit test for sinusoidal time