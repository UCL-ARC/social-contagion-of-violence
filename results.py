import os
import networkx as nx
import numpy as np

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti

import simulate_infections as si
import shuffle as st

runtime = 8 * 365  # simulation duration in days
baseline = 0.2  # prob that node becomes randomly infected
contagion = 0.11  # 0.14 #prob of triggering further infection (branching ratio)
self_contagion = False  # whether people can reinfect themselves
lifetime = 125  # average time between initial and triggered infection
n_nodes = 1000  # number of people
n_realizations = 10 #100
seed = 96


def params_todict(params):
    return {k: globals()[k] for k in params}


base_params_dict = params_todict(
    ['baseline', 'contagion', 'self_contagion', 'lifetime', 'runtime', 'n_nodes', 'n_realizations', 'seed'])
directory = si.set_results_directory(base_params_dict)

# Create a graph with varying number of edges, similar to social networks
g = nx.barabasi_albert_graph(n_nodes, 1, seed=seed)
adj_array = nx.to_numpy_array(g)
if self_contagion:
    adj_array = adj_array + np.eye(n_nodes)

min_edges = 3
_, pos = si.plot_network(g, min_edges=min_edges, filename='network.png',
                         params_dict=params_todict(['n_nodes', 'min_edges', 'seed']), directory=directory)

contagion_simu = SimuHawkesExpKernels(adjacency=contagion * adj_array,
                                      decays=1 / lifetime,
                                      baseline=baseline * np.ones(n_nodes) / runtime,
                                      seed=seed,
                                      end_time=runtime,
                                      verbose=True, )
contagion_simu.simulate()

contagion_timestamps = si.repeat_simulations(contagion_simu, n_realizations)

# There is a multi-threaded solution but it's slower on my environment:
# multi = SimuHawkesMulti(contagion_simu, n_realizations, n_threads=0)
# multi.simulate()
# contagion_timestamps = multi.timestamps

si.plot_simulations(contagion_timestamps, filename='contagion_multi_simulations.png',
                    params_dict=base_params_dict, directory=directory)

si.plot_homophily_variation(contagion_timestamps, g, filename='homophily_variation.png',
                            params_dict=base_params_dict, directory=directory)

si.set_homophily(contagion_simu.timestamps, g)
homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)

si.plot_homophily_network(g, pos=pos, filename='homophily_network.png',
                          params_dict={**base_params_dict, **{'homophily': homophily}}, directory=directory)

homophily_matrix = np.array([k for i, k in g.nodes.data('feature')])
homophily_simu = SimuHawkesExpKernels(adjacency=np.zeros((n_nodes, n_nodes)),
                                      decays=1 / lifetime,
                                      baseline=homophily_matrix * 0.5 * n_nodes / (sum(homophily_matrix) * runtime),
                                      seed=seed,
                                      end_time=runtime,
                                      verbose=True, )
homophily_simu.simulate()

homophily_timestamps = si.repeat_simulations(homophily_simu, n_realizations)
si.plot_simulations(homophily_timestamps, filename='homophily_multi_simulations.png',
                    params_dict={**base_params_dict, **{'homophily': homophily}}, directory=directory)

contagion_diffs = st.diff_neighbours(adj_array, contagion_simu.timestamps)
contagion_shuffled_diffs = st.repeat_shuffle_diff(adj_array, contagion_simu.timestamps, 100, seed=seed,
                                                  verbose=True)
st.plot_diff_comparison(contagion_diffs, contagion_shuffled_diffs, filename='contagion_diffs.png',
                        params_dict=base_params_dict, directory=directory)

homophily_diffs = st.diff_neighbours(adj_array, homophily_simu.timestamps)
homophily_shuffled_diffs = st.repeat_shuffle_diff(adj_array, homophily_simu.timestamps, 100, seed=seed,
                                                  verbose=True)
st.plot_diff_comparison(homophily_diffs, homophily_shuffled_diffs, filename='homophily_diffs.png',
                        params_dict={**base_params_dict, **{'homophily': homophily}}, directory=directory)
