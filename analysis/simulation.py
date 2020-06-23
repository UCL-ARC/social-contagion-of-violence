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
    contagion=0.11,  # prob of triggering further infection (branching ratio)
    self_contagion=True,  # whether people can reinfect themselves
    lifetime=125,  # average time between initial and triggered infection
    n_nodes=200,  # number of people
    n_realizations=5,  # how many times to repeat simulation
    seed=0,
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
an.plot_timestamps(contagion_simu.timestamps, filename='contagion_timestamps.png', directory=output_dir)
sh.plot_shuffle_test(adj_array, contagion_simu.timestamps, params['shuffles'], seed=params['seed'], verbose=True,
                     filename='contagion_shuffle_test.png', directory=output_dir)
sh.plot_shuffle_test(adj_array, contagion_simu.timestamps, params['shuffles'], smallest=True, seed=params['seed'],
                     verbose=True, filename='contagion_shuffle_test_smallest', directory=output_dir)

# Repeat simulation and investigate presence of homophily
# NOTE There is a multi-threaded solution but it's slower on my environment
contagion_timestamps = an.repeat_simulations(contagion_simu, params['n_realizations'])
an.plot_multi_timestamps(contagion_timestamps, filename='contagion_multi_simulations.png', directory=output_dir)
ho.plot_homophily_variation(contagion_timestamps, g, filename='contagion_assortativity_variation.png',
                            directory=output_dir)
ho.plot_homophily_variation(
    [sh.shuffle_timestamps(t, shuffle_nodes=True, seed=params['seed']) for t in contagion_timestamps],
    g, filename='contagion_assortativity_variation_shuffled', directory=output_dir)

########################################################################################################################
# HOMOPHILY

ho.set_homophily_random(g, values=[2], base=1, max_nodes=int(params['n_nodes'] * 0.3), seed=params['seed'])
ho.plot_homophily_network(g, pos=pos, filename='network_homophily', directory=output_dir)
homophily_baselines = ho.norm_time_functions(g, params['runtime'], 0.3)

homophily_simu = SimuHawkesExpKernels(adjacency=np.zeros((params['n_nodes'], params['n_nodes'])),
                                      decays=0,
                                      baseline=homophily_baselines,
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
ho.plot_homophily_variation(
    [sh.shuffle_timestamps(t, shuffle_nodes=True, seed=params['seed']) for t in homophily_timestamps],
    g, filename='homophily_assortativity_variation_shuffled', directory=output_dir)

########################################################################################################################
# INHOMOGENEOUS

t_values = np.array([0, params['lifetime'], params['runtime'] * 0.5, params['runtime'] * 0.5 + params['lifetime']])
y_values = np.array([0.5 / params['lifetime'], 0, 0.5 / params['lifetime'], 0]) * 0.3
tf = TimeFunction((t_values, y_values), inter_mode=TimeFunction.InterConstRight)
ho.plot_time_functions(tf, filename='Timefunction_Inhomogeneous', directory=output_dir)
inhomogeneous_baselines = [tf for i in range(params['n_nodes'])]

inhomogeneous_simu = SimuHawkesExpKernels(adjacency=np.zeros([params['n_nodes'], params['n_nodes']]),
                                          decays=0,
                                          end_time=params['runtime'],
                                          baseline=inhomogeneous_baselines,
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
ho.plot_homophily_variation(
    [sh.shuffle_timestamps(t, shuffle_nodes=True, seed=params['seed']) for t in inhomogeneous_timestamps], g,
    filename='inhomogeneous_assortativity_variation_shuffled', directory=output_dir)

########################################################################################################################
# CONFOUNDING

ho.set_homophily_random(g, max_nodes=int(params['n_nodes'] * 0.3), seed=params['seed'])
ho.plot_homophily_network(g, pos=pos, filename='network_confounding', directory=output_dir)
confounding_baselines = ho.peak_time_functions(g, params['runtime'], params['lifetime'],base=0.1)
ho.plot_time_functions(confounding_baselines, filename='Timefunctions_counfounding', directory=output_dir)

confounding_simu = SimuHawkesExpKernels(adjacency=np.zeros([params['n_nodes'], params['n_nodes']]),
                                        decays=0,
                                        end_time=params['runtime'],
                                        baseline=confounding_baselines,
                                        seed=params['seed'],
                                        verbose=True)
confounding_simu.simulate()
an.plot_timestamps(confounding_simu.timestamps, filename='confounding_timestamps.png', directory=output_dir,
                   ax1_kw=dict(bins=50))
sh.plot_shuffle_test(adj_array, confounding_simu.timestamps, params['shuffles'], smallest=False, seed=params['seed'],
                     verbose=True, filename='confounding_shuffle_test.png', directory=output_dir)
sh.plot_shuffle_test(adj_array, confounding_simu.timestamps, params['shuffles'], smallest=True, seed=params['seed'],
                     verbose=True, filename='confounding_shuffle_test_smallest.png', directory=output_dir)

# Repeat simulation and investigate presence of homophily
confounding_timestamps = an.repeat_simulations(confounding_simu, params['n_realizations'])
an.plot_multi_timestamps(confounding_timestamps, filename='confounding_multi_simulations.png', directory=output_dir)
ho.plot_homophily_variation(confounding_timestamps, g, filename='confounding_assortativity_variation.png',
                            directory=output_dir)
ho.plot_homophily_variation(
    [sh.shuffle_timestamps(t, shuffle_nodes=True, seed=params['seed']) for t in confounding_timestamps], g,
    filename='confounding_assortativity_variation_shuffled', directory=output_dir)

########################################################################################################################
## SAVE

# Remote
# mlflow.set_tracking_uri('databricks')
# mlflow.set_experiment('/Users/soumaya.mauthoor.17@ucl.ac.uk/contagion_simulation')
#
# Locally
# mlflow.set_tracking_uri('./mlruns')
# mlflow.set_experiment('/simulation')
#
# with mlflow.start_run():
#     mlflow.log_params(params)
#     mlflow.log_artifacts(output_dir)
