import numpy as np
from scipy.optimize import minimize, basinhopping, brute
import networkx as nx
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels
import src.inference as infer
import src.analyse as an

#######################################################################################################################
# INITIALISE

params = dict(
    runtime=2000,  # simulation duration in days
    baseline=0.2,  # expected number of times node becomes randomly infected over runtime
    contagion=0.1,  # expected number of triggered infections (branching ratio)
    self_contagion=False,  # whether people can reinfect themselves
    lifetime=5,  # average time between initial and triggered infection
    n_nodes=500,  # number of people
    n_realizations=10,  # how many times to repeat simulation
    seed=0,
)

# Create a graph with varying number of edges, similar to social networks
g = nx.barabasi_albert_graph(params['n_nodes'], 1, seed=params['seed'])
# g = nx.newman_watts_strogatz_graph(params['n_nodes'], 2, 0.2)
adj_array = nx.to_numpy_array(g)

#######################################################################################################################
# CONSTANT BASELINE

simu = SimuHawkesExpKernels(adjacency=params['contagion'] * adj_array,
                            decays=1 / params['lifetime'],
                            baseline=params['baseline'] * np.ones(params['n_nodes']) / params['runtime'],
                            seed=params['seed'],
                            end_time=params['runtime'],
                            verbose=True, )
simu.simulate()
timestamps = an.repeat_simulations(simu, params['n_realizations'])
print([len(np.concatenate(t)) for t in timestamps])
# [141, 139, 190, 155, 129, 170, 110, 201, 145, 140]

rranges = (slice(1e-10, 1 / params['runtime'], 0.2 / params['runtime']),
           slice(0.01, 0.2, 0.04),
           slice(1e-10, 1, 0.2))
coeffs = np.zeros([params['n_realizations'], 3])
actual = np.array([params['baseline'] / params['runtime'], params['contagion'], 1 / params['lifetime']])

for i, t in enumerate(timestamps):
    brute_res = brute(infer.crit_multi, ranges=rranges, args=(g, t, params['runtime'], 0, 1, 0, False),
                      full_output=True, finish=0, )
    ggd_res = minimize(infer.crit_multi, x0=brute_res[0], args=(g, t, params['runtime'], 0, 1, 0, False),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, 1), (1e-10, None)))
    print(ggd_res.x)
    coeffs[i] = (actual - ggd_res.x) / actual

print(coeffs * 100)
# [[  2.71529476  13.77429743   4.19380027]
#  [ -5.91931876   8.69790527  -9.88089247]
#  [ -3.92955215  -3.81932082 -42.44814746]
#  [ -9.2046291    8.99386984 -14.05709364]
#  [ 11.3687572    8.89775856 -11.90291022]
#  [  5.079783    -4.47609715  23.44144741]
#  [ 21.70820886  10.12619028 -12.16861821]
#  [ 22.00248649  -8.18181947   2.04373841]
#  [ -8.51042326   8.96097917 -12.08272216]
#  [ -7.76254987  10.03352941  -0.94549283]]

print(np.average(coeffs*100,0))
# [ 2.75480572  5.30072925 -7.38068909]
print(np.std(coeffs*100,0))
# [11.44456751  7.27696873 16.00075118]