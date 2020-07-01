import numpy as np
from scipy.optimize import minimize, basinhopping, brute
import networkx as nx
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels, HawkesExpKern
from tick.base import TimeFunction
import src.inference as infer
import src.homophily as ho

#######################################################################################################################
# INITIALISE

params = dict(
    runtime=8 * 365,  # simulation duration in days
    baseline=10,  # prob that node becomes randomly infected
    contagion=0.12,  # prob of triggering further infection (branching ratio)
    self_contagion=False,  # whether people can reinfect themselves
    lifetime=125,  # average time between initial and triggered infection
    n_nodes=1000,  # number of people
    n_realizations=5,  # how many times to repeat simulation
    seed=3,
    shuffles=100,  # how many times to shuffle experiment
)

# Create a graph with varying number of edges, similar to social networks
g = nx.barabasi_albert_graph(params['n_nodes'], 1, seed=params['seed'])
adj_array = nx.to_numpy_array(g)

#######################################################################################################################
# CONSTANT BASELINE

contagion_simu = SimuHawkesExpKernels(adjacency=params['contagion'] * adj_array,
                                      decays=1 / params['lifetime'],
                                      baseline=params['baseline'] * np.ones(params['n_nodes']) / params['runtime'],
                                      seed=params['seed'],
                                      end_time=params['runtime'],
                                      verbose=True, )
contagion_simu.simulate()
t = contagion_simu.timestamps

np.arange(1e-10, 50 / params['runtime'], 10 / params['runtime'])
np.arange(0.01, 0.2, 0.04)
np.arange(1 / params['runtime'], 50 / params['runtime'], 10 / params['runtime'])
# Brute Search
rranges = (slice(1e-10, 50 / params['runtime'], 10 / params['runtime']),
           slice(0.01, 0.2, 0.04),
           slice(1 / params['runtime'], 50 / params['runtime'], 10 / params['runtime']))
brute_res = brute(infer.crit_multi, ranges=rranges, args=(g, t, params['runtime'], 0, 1, 0, True),
                  full_output=True, finish=0, )
print(brute_res[0])
# [0.00342466 0.13       0.00719178]

# global gradient descent
ggd_res = minimize(infer.crit_multi, x0=brute_res[0], args=(g, t, params['runtime'], 0, 1, 0, False), method='L-BFGS-B',
                   bounds=((1e-10, None), (1e-10, None), (1e-10, None)))
print(ggd_res.x)
# [0.00365751 0.11405178 0.00861958]

print(-infer.log_likelihood_multi(g, t, params['baseline'] / params['runtime'],
                                  params['contagion'], 1 / params['lifetime'], params['runtime']))
print(-infer.log_likelihood_multi(g, t, *brute_res[0], params['runtime']))
print(-infer.log_likelihood_multi(g, t, *ggd_res.x, params['runtime']))
print(100 * (np.array(
    [params['baseline'] / params['runtime'], params['contagion'], 1 / params['lifetime']]) - ggd_res.x) / ggd_res.x)
# 121523.11901323672
# 121558.66157932763
# 121509.45915473448
# [-6.36634018  5.21536497 -7.18806359]