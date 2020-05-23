import numpy as np
import simulate_infections as si
import pytest
import os

os.environ['DISPLAY'] = '0'
from tick.hawkes import SimuHawkesExpKernels
import networkx as nx

data_estimate_event_counts = [([[0]], 0, (200.0, 0.0)),
                              ([[1]], 0.2, (250.0, 50.0)),
                              ([[0, 0], [0, 0]], 0, (400.0, 0.0)),
                              ([[1, 0], [0, 1]], 0.2, (500.0, 100.0)),
                              ([[0, 1], [1, 0]], 0.2, (500.0, 100.0)),
                              ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 0.2, (750.0, 150.0)),
                              ]


# end_time = 100
# base = 2
# contagion = 0.2
# ratio = contagion / (1 - contagion); print(ratio)
#
# adj = [[1, 1], [1, 1]]
# adj = [[0,1],[1,0]]
# adj = [[1, 0, 0],[0, 1, 0],[0, 0, 1]]
# adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
# adj = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
# adj = np.eye(4,k=1)+np.eye(4,k=-1) ; g=si._graph(adj); g.add_edge(0,3); adj=nx.to_numpy_array(g)
# g=nx.Graph();g.add_edges_from([(0,1),(0,2),(0,3), (0,4)]); adj=nx.to_numpy_array(g)
# adj = [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0] ]
#
# si.plot_network(g,0)
# i = 2; nnx = np.linalg.matrix_power(adj,i) ; print(nnx)
# contagions = base * end_time * sum(nnx*ratio**(i+1)); print(contagions)


@pytest.mark.parametrize("adj,contagion,expected", data_estimate_event_counts, ids=str)
def test_estimate_event_counts_high_base_low_nodes(adj, contagion, expected):
    end_time = 100
    base = 2
    counts = si.estimate_event_counts(adj=np.array(adj), base=base, end_time=end_time, alpha=contagion, nn=1)
    assert counts == expected

    s = SimuHawkesExpKernels(adjacency=np.array(adj) * contagion, baseline=np.ones(len(adj)) * base,
                             decays=1, seed=1, end_time=end_time, verbose=False)
    r = si.repeat_simulations(s, n_simulations=100)
    avg = np.average([len(np.concatenate(t)) for t in r])
    std = np.std([len(np.concatenate(t)) for t in r])
    for n in range(len(adj)):
        print(np.average([len(t[n]) for t in r]) - base * end_time)
    print('Average events per simulation: ', avg)
    print('Standard deviation of events: ', std)
    assert avg - std <= expected[0] <= avg + std


def test_set_homophily():
    g = nx.Graph();
    g.add_edges_from([(0, 1), (0, 2)])
    timestamps = [np.array([]), np.array([1, 2]), np.array([1])]
    si.set_homophily(timestamps, g)
    assert [1, 2, 2] == [k for i, k in g.nodes.data('feature')]


def test_set_homophily_node_list():
    g = nx.Graph();
    g.add_edges_from([(0, 1), (0, 3)])
    timestamps = [np.array([]), np.array([1, 2]), np.array([1])]
    si.set_homophily(timestamps, g , node_list=(0,1,3))
    assert [1, 2, 2] == [k for i, k in g.nodes.data('feature')]
