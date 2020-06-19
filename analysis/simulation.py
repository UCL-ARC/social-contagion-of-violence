import networkx as nx
import numpy as np
import os
import mlflow

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti
from tick.base import TimeFunction

import src.analyse as an
import src.shuffle as sh
import src.homophily as ho
import src.common as co

########################################################################################################################
# INITIALISE

params = dict(
    runtime=8 * 365,  # simulation duration in days
    baseline=0.2,  # prob that node becomes randomly infected
    contagion=0.11,  # 0.14 # prob of triggering further infection (branching ratio)
    self_contagion=False,  # whether people can reinfect themselves
    lifetime=125,  # average time between initial and triggered infection
    n_nodes=1000,  # number of people
    n_realizations=5,  # how many times to repeat simulation
    seed=3,  # 96
    shuffles=100,  # how many times to shuffle experiment
)
output_dir = co.set_directory('analysis/results/simulation/')

# Create a graph with varying number of edges, similar to social networks
g = nx.barabasi_albert_graph(params['n_nodes'], 1, seed=params['seed'])
adj_array = nx.to_numpy_array(g)
if params['self_contagion']:
    adj_array = adj_array + np.eye(params['n_nodes'])

_, pos = an.plot_network(g, min_edges=3, filename='network.png', directory=output_dir)

########################################################################################################################
# CONTAGION

contagion_simu = SimuHawkesExpKernels(adjacency=params['contagion'] * adj_array,
                                      decays=1 / params['lifetime'],
                                      baseline=params['baseline'] * np.ones(params['n_nodes']) / params['runtime'],
                                      seed=params['seed'],
                                      end_time=params['runtime'],
                                      verbose=True, )
contagion_simu.simulate()
ts = contagion_simu.timestamps
an.plot_timestamps(ts, filename='contagion_timestamps.png', directory=output_dir)
sh.plot_shuffle_test(adj_array, ts, params['shuffles'], seed=params['seed'], verbose=True,
                     filename='contagion_shuffle_test.png', directory=output_dir)
sh.plot_shuffle_test(adj_array, ts, params['shuffles'], smallest=True, seed=params['seed'],
                     verbose=True, filename='contagion_shuffle_test_smallest', directory=output_dir)

sh.plot_shuffle_test(adj_array, ts, params['shuffles'], shuffle_nodes=True, seed=params['seed'], verbose=True, )
sh.plot_shuffle_test(adj_array, ts, params['shuffles'], shuffle_nodes=True,smallest=True,
                     seed=params['seed'], verbose=True, )

# Repeat simulation and investigate presence of homophily
contagion_timestamps = an.repeat_simulations(contagion_simu, params['n_realizations'])

# There is a multi-threaded solution but it's slower on my environment:
# multi = SimuHawkesMulti(contagion_simu, n_realizations, n_threads=0)
# multi.simulate()
# contagion_timestamps = multi.timestamps

an.plot_multi_timestamps(contagion_timestamps, filename='contagion_multi_simulations.png', directory=output_dir)
ho.plot_homophily_variation(contagion_timestamps, g, filename='contagion_assortativity_variation.png',
                            directory=output_dir)
ho.plot_homophily_variation([sh.shuffle_timestamps(t) for t in contagion_timestamps], g,
                            filename='contagion_assortativity_variation_shuffled', directory=output_dir)

########################################################################################################################
# HOMOPHILY

ho.set_homophily(ts, g)
homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)

ho.plot_homophily_network(g, pos=pos, filename='network_homophily.png', directory=output_dir)

homophily_matrix = np.array([k for i, k in g.nodes.data('feature')])
homophily_simu = SimuHawkesExpKernels(adjacency=np.zeros((params['n_nodes'], params['n_nodes'])),
                                      decays=1 / params['lifetime'],
                                      baseline=homophily_matrix * 0.5 * params['n_nodes']
                                               / (sum(homophily_matrix) * params['runtime']),
                                      seed=params['seed'],
                                      end_time=params['runtime'],
                                      verbose=True, )
homophily_simu.simulate()
an.plot_timestamps(homophily_simu.timestamps, filename='homophily_timestamps.png', directory=output_dir)
sh.plot_shuffle_test(adj_array, homophily_simu.timestamps, params['shuffles'], smallest=False,
                     seed=params['seed'], verbose=True, filename='homophily_shuffle_test.png', directory=output_dir)
sh.plot_shuffle_test(adj_array, homophily_simu.timestamps, params['shuffles'], smallest=True, seed=params['seed'],
                     verbose=True, filename='homophily_shuffle_test_smallest.png', directory=output_dir)

# Repeat simulation and investigate presence of homophily
homophily_timestamps = an.repeat_simulations(homophily_simu, params['n_realizations'])
an.plot_multi_timestamps(homophily_timestamps, filename='homophily_multi_simulations.png', directory=output_dir)
ho.plot_homophily_variation(homophily_timestamps, g, filename='homophily_assortativity_variation.png',
                            directory=output_dir)
ho.plot_homophily_variation([sh.shuffle_timestamps(t) for t in homophily_timestamps], g,
                            filename='homophily_assortativity_variation_shuffled', directory=output_dir)

########################################################################################################################
# INHOMOGENEOUS

t_values = np.array([0, params['lifetime'], params['runtime'] * 0.5, params['runtime'] * 0.5 + params['lifetime']])
y_values = np.array([0.25 / params['lifetime'], 0, 0.25 / params['lifetime'], 0])
tf = TimeFunction((t_values, y_values), inter_mode=TimeFunction.InterConstRight)
from tick.plot import plot_timefunction

plot_timefunction(tf)
baselines = [tf for i in range(params['n_nodes'])]

inhomogeneous_simu = SimuHawkesExpKernels(adjacency=np.zeros([params['n_nodes'], params['n_nodes']]),
                                          decays=0,
                                          end_time=params['runtime'],
                                          baseline=baselines,
                                          seed=params['seed'],
                                          verbose=True)
inhomogeneous_simu.simulate()
an.plot_timestamps(inhomogeneous_simu.timestamps, filename='inhomogeneous_timestamps.png', directory=output_dir,
                   ax1_kw=dict(bins=50))
sh.plot_shuffle_test(adj_array, inhomogeneous_simu.timestamps, params['shuffles'], smallest=False, seed=params['seed'],
                     verbose=True, filename='inhomogeneous_shuffle_test.png', directory=output_dir)
sh.plot_shuffle_test(adj_array, inhomogeneous_simu.timestamps, params['shuffles'], smallest=True, seed=params['seed'],
                     verbose=True, filename='inhomogeneous_shuffle_test_smallest.png', directory=output_dir)

# Repeat simulation and investigate presence of homophily
inhomogeneous_timestamps = an.repeat_simulations(inhomogeneous_simu, params['n_realizations'])
an.plot_multi_timestamps(inhomogeneous_timestamps, filename='inhomogeneous_multi_simulations.png', directory=output_dir)
ho.plot_homophily_variation(inhomogeneous_timestamps, g, filename='inhomogeneous_assortativity_variation.png',
                            directory=output_dir)
ho.plot_homophily_variation([sh.shuffle_timestamps(t) for t in inhomogeneous_timestamps], g,
                            filename='inhomogeneous_assortativity_variation_shuffled', directory=output_dir)

########################################################################################################################
## SAVE

# Remote
# mlflow.set_tracking_uri('databricks')
# mlflow.set_experiment('/Users/soumaya.mauthoor.17@ucl.ac.uk/contagion_simulation')
#
# # Locally
# mlflow.set_tracking_uri('./mlruns')
# mlflow.set_experiment('/simulation')
#
# with mlflow.start_run():
#     mlflow.log_params(params)
#     mlflow.log_artifacts(output_dir)
