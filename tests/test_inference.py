import numpy as np
import src.inference as infer
import pytest
import networkx as nx

mu = 0.5
alpha = 0.2
beta = np.log(2)
runtime = 4
timestamps = np.array([0, 2, 3])
timestamps_1 = np.array([0, 1, 2])
timestamps_2 = np.array([])
timestamps_neighbors = [timestamps_1, timestamps_2]

# Expected recursive array:
r0 = 0
r1 = np.exp(-beta * (timestamps[1] - 0))
r2 = np.exp(-beta * (timestamps[2] - timestamps[1])) * (1 + r1)
r_array = np.array([r0, r1, r2])

# Expected multi-variate recursive array:
r_array_multi = np.zeros((len(timestamps), len(timestamps_neighbors)))
for l in range(1, len(timestamps)):
    for n in range(len(timestamps_neighbors)):
        ts = np.where(timestamps_neighbors[n] < timestamps[l])
        r_array_multi[l, n] = np.sum(np.exp(-beta * (timestamps[l] - timestamps_neighbors[n][ts])))


def test_recursive():
    np.testing.assert_array_equal(infer._recursive(timestamps, beta), r_array)


def test_recursive_multi():
    np.testing.assert_array_equal(infer._recursive_multi(timestamps, timestamps_neighbors, beta), r_array_multi)


def test_loglikelihood():
    expected = -mu * runtime - alpha * np.sum(1 - np.exp(-beta * (runtime - timestamps))) + \
               np.sum(np.log(mu + alpha * beta * r_array))
    assert infer.log_likelihood(timestamps, mu, alpha, beta, runtime) == pytest.approx(expected)


def test_loglikelihood_multi():
    integral = -3 * mu * runtime \
               - 2 * alpha * np.sum(1 - np.exp(-beta * (runtime - timestamps))) \
               - alpha * np.sum(1 - np.exp(-beta * (runtime - timestamps_1))) \
               + np.sum(np.log(mu + alpha * beta * np.sum(r_array_multi,1))) \
               + np.sum(np.log(mu + alpha * beta *
                               np.sum(infer._recursive_multi(timestamps_1, [timestamps], beta), 1)))
               # node 2 doesn't have contribution because doesn't have any timestamps
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 2)])
    actual = infer.log_likelihood_multi(g, [timestamps, timestamps_1, timestamps_2], mu, alpha, beta, runtime)
    assert integral == actual