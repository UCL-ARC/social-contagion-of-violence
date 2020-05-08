import simulate_infections as si
import os
import networkx as nx
import numpy as np
os.environ['DISPLAY'] = '0' #Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti

baseline = 0.0002
contagion = 0.14  # 0.11
self_contagion = True
lifetime = 7
runtime = 1000
n_nodes = 100  # 1000
n_realizations = 100  # 20
seed = 1


def params_todict(params):
    return {k: globals()[k] for k in params}


base_params_dict = params_todict(
    ['baseline', 'contagion', 'self_contagion', 'lifetime', 'runtime', 'n_nodes', 'n_realizations', 'seed'])
directory = si.set_results_directory(base_params_dict)

# Create a graph with varying number of edges, similar to social networks
g = nx.barabasi_albert_graph(n_nodes, 1, seed=seed)

# Create a graph with exactly two edges, useful for testing purposes
# adjacency = np.eye(n_nodes,k=1) + np.eye(n_nodes,k=-1)
# g = nx.from_numpy_array(adjacency)

min_edges = 3
_, pos = si.plot_graph(g, min_edges=min_edges, filename='network.png',
                    params_dict=params_todict(['n_nodes', 'min_edges', 'seed']), directory=directory)
si.estimate_event_counts(g, baseline, runtime, contagion, n_nodes, nn=30)

self_contagion_matrix = np.eye(n_nodes) if self_contagion else np.zeros((n_nodes, n_nodes))
contagion_simulation = SimuHawkesExpKernels(adjacency=contagion * (nx.to_numpy_array(g) + self_contagion_matrix),
                                            decays=1 / lifetime,
                                            baseline=baseline * np.ones(n_nodes),
                                            seed=seed,
                                            end_time=runtime,
                                            verbose=True, )
contagion_simulation.simulate()

contagion_timestamps = si.repeat_simulations(contagion_simulation, n_realizations)

# There is a multi-threaded solution but it's slower on my environment:
# multi = SimuHawkesMulti(contagion_simulation, n_realizations, n_threads=0)
# multi.simulate()
# contagion_timestamps = multi.timestamps

si.plot_simulations(contagion_timestamps, filename='contagion_multi_simulations.png',
                 params_dict=base_params_dict, directory=directory)

si.plot_homophily_variation(contagion_timestamps, g, filename='homophily_variation.png',
                         params_dict=base_params_dict, directory=directory)

si.set_homophily(contagion_simulation.timestamps, g)
homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)

si.plot_homophily_network(g, pos=pos, filename='homophily_network.png',
                       params_dict={**base_params_dict, **{'homophily': homophily}}, directory=directory)

homophily_matrix = np.array([k for i, k in g.nodes.data('feature')])
homophily_simulation = SimuHawkesExpKernels(adjacency=np.zeros((n_nodes, n_nodes)),
                                            decays=1 / lifetime,
                                            baseline=homophily_matrix * 0.5 * n_nodes / (
                                                    sum(homophily_matrix) * runtime),
                                            seed=seed,
                                            end_time=runtime,
                                            verbose=True, )
homophily_simulation.simulate()

homophily_timestamps = si.repeat_simulations(homophily_simulation, n_realizations)
si.plot_simulations(homophily_timestamps, filename='homophily_multi_simulations.png',
                 params_dict={**base_params_dict, **{'homophily': homophily}}, directory=directory)
