import networkx as nx
import numpy as np
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti
from tick.base import TimeFunction

import src.analyse as an
import src.shuffle as sh
import src.homophily as ho
import src.common as co

########################################################################################################################
# DATA

runtime = 8 * 365  # simulation duration in days
baseline = 0.2  # prob that node becomes randomly infected
contagion = 0.11  # 0.14 #prob of triggering further infection (branching ratio)
self_contagion = False  # whether people can reinfect themselves
lifetime = 125  # average time between initial and triggered infection
n_nodes = 1000  # number of people
n_realizations = 10  # 100
seed = 3  # 96


def params_todict(params):
    return {k: globals()[k] for k in params}


base_params_dict = params_todict(
    ['baseline', 'contagion', 'self_contagion', 'lifetime', 'runtime', 'n_nodes', 'n_realizations', 'seed'])
directory = co.set_directory('result/simulation/' + co.dict_string(base_params_dict))

# Create a graph with varying number of edges, similar to social networks
g = nx.barabasi_albert_graph(n_nodes, 1, seed=seed)
adj_array = nx.to_numpy_array(g)
if self_contagion:
    adj_array = adj_array + np.eye(n_nodes)

min_edges = 3
_, pos = an.plot_network(g, min_edges=min_edges, filename='network.png',
                         params_dict=params_todict(['n_nodes', 'min_edges', 'seed']), directory=directory)

contagion_simu = SimuHawkesExpKernels(adjacency=contagion * adj_array,
                                      decays=1 / lifetime,
                                      baseline=baseline * np.ones(n_nodes) / runtime,
                                      seed=seed,
                                      end_time=runtime,
                                      verbose=True, )
contagion_simu.simulate()
an.plot_timestamps(contagion_simu.timestamps, filename='timestamps.png',
                   params_dict=base_params_dict, directory=directory)

########################################################################################################################
# HOMOPHILY

contagion_timestamps = an.repeat_simulations(contagion_simu, n_realizations)

# There is a multi-threaded solution but it's slower on my environment:
# multi = SimuHawkesMulti(contagion_simu, n_realizations, n_threads=0)
# multi.simulate()
# contagion_timestamps = multi.timestamps

an.plot_multi_timestamps(contagion_timestamps, filename='contagion_multi_simulations.png',
                         params_dict=base_params_dict, directory=directory)

ho.plot_homophily_variation(contagion_timestamps, g, filename='homophily_variation.png',
                            params_dict=base_params_dict, directory=directory)

ho.set_homophily(contagion_simu.timestamps, g)
homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)

ho.plot_homophily_network(g, pos=pos, filename='homophily_network.png',
                          params_dict={**base_params_dict, **{'homophily': homophily}}, directory=directory)

homophily_matrix = np.array([k for i, k in g.nodes.data('feature')])
homophily_simu = SimuHawkesExpKernels(adjacency=np.zeros((n_nodes, n_nodes)),
                                      decays=1 / lifetime,
                                      baseline=homophily_matrix * 0.5 * n_nodes / (sum(homophily_matrix) * runtime),
                                      seed=seed,
                                      end_time=runtime,
                                      verbose=True, )
homophily_simu.simulate()

homophily_timestamps = an.repeat_simulations(homophily_simu, n_realizations)
an.plot_multi_timestamps(homophily_timestamps, filename='homophily_multi_simulations.png',
                         params_dict={**base_params_dict, **{'homophily': homophily}}, directory=directory)

########################################################################################################################
# SHUFFLE

contagion_diffs = sh.diff_neighbours(adj_array, contagion_simu.timestamps)
shuffles = 100
contagion_shuffled_diffs = sh.repeat_shuffle_diff(adj_array, contagion_simu.timestamps, shuffles, seed=seed,
                                                  verbose=True)
sh.plot_diff_comparison(contagion_diffs, contagion_shuffled_diffs, filename='contagion_diffs.png',
                        params_dict={**base_params_dict, **{'shuffles': shuffles}}, directory=directory)

homophily_diffs = sh.diff_neighbours(adj_array, homophily_simu.timestamps)
homophily_shuffled_diffs = sh.repeat_shuffle_diff(adj_array, homophily_simu.timestamps, shuffles, seed=seed,
                                                  verbose=True)
sh.plot_diff_comparison(homophily_diffs, homophily_shuffled_diffs, filename='homophily_diffs.png', directory=directory,
                        params_dict={**base_params_dict, **{'homophily': homophily, 'shuffles': shuffles}})

########################################################################################################################
# INHOMOGENEOUS

from tick.plot import plot_point_process

# n_nodes = 1
t_values = np.linspace(0, lifetime)
y_values = np.maximum(np.sin(t_values * (2 * np.pi) / lifetime), 0) * 5 / runtime
t_values = np.linspace(0, lifetime * 8)
y_values = y_values[0::4]
y_values = np.concatenate([y_values, np.zeros([37])])
tf = TimeFunction((t_values, y_values), border_type=TimeFunction.Cyclic)
baselines = [tf for i in range(n_nodes)]
decay = 0
adjacency = np.zeros([n_nodes, n_nodes])

inhomogeneous_simu = SimuHawkesExpKernels(adjacency, decay, end_time=runtime, baseline=baselines, seed=seed,
                                          verbose=False)
# inhomogeneous_simu.track_intensity(0.1)
inhomogeneous_simu.simulate()
# plot_point_process(inhomogeneous_simu, show=True)

an.plot_timestamps(inhomogeneous_simu.timestamps, filename='inhomogeneous_timestamps.png', directory=directory,
                   params_dict=params_todict(['n_nodes','runtime','seed','decay']))

ih_diffs = sh.diff_neighbours(adj_array, inhomogeneous_simu.timestamps)
ih_shuffled_diffs = sh.repeat_shuffle_diff(adj_array, inhomogeneous_simu.timestamps, shuffles, seed=seed, verbose=True)

sh.plot_diff_comparison(ih_diffs,ih_shuffled_diffs, filename='inhomogeneous_diffs.png', directory=directory,
                   params_dict=params_todict(['n_nodes','seed']))

