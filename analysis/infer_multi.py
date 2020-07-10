import numpy as np
from scipy.optimize import minimize, basinhopping, brute
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from mord import LogisticAT
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels
from tick.plot import plot_point_process
import src.inference as infer
import src.analyse as an
import src.homophily as ho

#######################################################################################################################
# INITIALISE

params = dict(
    runtime=150,  # simulation duration in days
    prediction_time=10,  # number of days to predict simulation over
    baseline=0.3,  # expected number of times node becomes randomly infected over runtime
    contagion=0.1,  # expected number of "direct" triggered infections
    self_contagion=False,  # whether people can reinfect themselves
    total_rate=0.5,  # expected total number of infections per node over runtime
    lifetime=5,  # average time between initial and triggered infection
    n_nodes=1000,  # number of people
    n_realizations=10,  # how many times to repeat simulation
    seed=1,
)

# Create a graph with varying number of edges, similar to social networks
g = nx.barabasi_albert_graph(params['n_nodes'], 1, seed=params['seed'])
# g = nx.newman_watts_strogatz_graph(params['n_nodes'], 2, 0.2)
adj_array = nx.to_numpy_array(g)

coeffs_range = (slice(1e-10, 1 / params['runtime'], 0.2 / params['runtime']),  # mu range
                slice(1e-10, 0.5, 0.1),  # alpha range
                slice(1e-10, 0.5, 0.1))  # beta range
coeffs_actual = np.array([params['baseline'] / params['runtime'], params['contagion'], 1 / params['lifetime']])
#######################################################################################################################
# SIMPLE CONTAGION

contagion_simu = SimuHawkesExpKernels(adjacency=params['contagion'] * adj_array,
                                      decays=1 / params['lifetime'],
                                      baseline=params['baseline'] * np.ones(params['n_nodes']) / params['runtime'],
                                      seed=params['seed'],
                                      end_time=params['runtime'] + params['prediction_time'],
                                      verbose=True)
contagion_ts = an.repeat_simulations(contagion_simu, params['n_realizations'])
print([len(np.concatenate(i)) for i in contagion_ts])
# [467, 538, 494, 564, 594, 471, 577, 527, 506, 492]
an.plot_multi_timestamps(contagion_ts)

contagion_coeffs_error = np.zeros([params['n_realizations'], 3])
contagion_scores = np.zeros([params['n_realizations'], params['prediction_time'] + 1])
contagion_comparison_scores = np.zeros([params['n_realizations'], params['prediction_time']])
for i, t_all in enumerate(contagion_ts):
    t, t_right = an.split_timestamps(t_all, params['runtime'])
    brute_res = brute(infer.crit_multi, ranges=coeffs_range, args=(g, t, params['runtime'], 0, 1, 0, False),
                      full_output=True, finish=0, )
    ggd_res = minimize(infer.crit_multi, x0=brute_res[0], args=(g, t, params['runtime'], 0, 1, 0, False),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, 1), (1e-10, None)))
    print(ggd_res.x)
    contagion_coeffs_error[i] = (coeffs_actual - ggd_res.x) / coeffs_actual

    contagion_risk = infer.contagion_risk(g, t_all, *ggd_res.x[1:],
                                          range(params['runtime'], params['runtime'] + params['prediction_time']))
    predicted_nodes = np.array([infer.get_highest_risk_nodes(day) for day in contagion_risk])
    infected_nodes = an.get_infected_nodes(t_all,
                                           range(params['runtime'], params['runtime'] + params['prediction_time']))
    rng = np.random.default_rng(i)
    comparison_nodes = rng.binomial(n=1, p=0.01, size=[params['prediction_time'], g.number_of_nodes()])

    for v, (a, b, c) in enumerate(zip(infected_nodes, predicted_nodes, comparison_nodes)):
        contagion_scores[i, v] = confusion_matrix(a, b)[1, 1]
        contagion_comparison_scores[i, v] = confusion_matrix(a, c)[1, 1]
    contagion_scores[i, -1] = np.sum(infected_nodes)

print(np.average(contagion_coeffs_error, 0) * 100)
print(np.std(contagion_coeffs_error, 0) * 100)
# [-2.69751519  1.74550079 -2.727759  ]
# [ 5.80736152  6.35454599 15.44353994]

print(np.average(np.sum(contagion_scores[:, :-1], 1) / contagion_scores[:, -1]) * 100)
print(np.average(np.sum(contagion_comparison_scores[:, :-1], 1) / contagion_scores[:, -1]) * 100)
# 8.949255611269569
# 0.37037037037037035
#######################################################################################################################
# DEMOGRAPHIC INFECTION

rng = np.random.default_rng(params['seed'])
gender = rng.choice((-1, 1), size=params['n_nodes'], p=(0.9, 0.1))
age = rng.choice(range(-2, 2), size=params['n_nodes'], p=(0.4, 0.3, 0.2, 0.1))
arrests = rng.choice(range(-2, 2), size=params['n_nodes'], p=(0.4, 0.3, 0.2, 0.1))
demographic = gender + age + arrests
demographic = 1 / (1 + np.exp(-demographic))
demographic = params['total_rate'] * demographic * params['n_nodes'] / sum(demographic)
x = np.array([gender, age, arrests]).T

random_simu = SimuHawkesExpKernels(adjacency=np.zeros((params['n_nodes'], params['n_nodes'])),
                                   decays=0,
                                   baseline=demographic / params['runtime'],
                                   seed=params['seed'],
                                   end_time=params['runtime'] + params['prediction_time'],
                                   verbose=True, )
random_ts = an.repeat_simulations(random_simu, params['n_realizations'])
an.plot_multi_timestamps(random_ts)
print([len(np.concatenate(i)) for i in random_ts])
# [557, 521, 524, 586, 535, 525, 538, 525, 535, 549]

random_coeffs = np.zeros([params['n_realizations'], 4])
random_scores = np.zeros([params['n_realizations'], params['prediction_time'] + 1])
random_comparison_scores = np.zeros([params['n_realizations'], params['prediction_time']])
for i, t_all in enumerate(random_ts):
    t, t_right = an.split_timestamps(t_all, params['runtime'])

    ho.set_feature_timestamps(g, t)
    y = np.array(list(nx.get_node_attributes(g, 'feature').values()))
    model = LogisticRegression(random_state=params['seed'])
    model.fit(x, y)
    random_coeffs[i] = np.concatenate([model.coef_[0], model.intercept_])

    baseline_pred = model.predict_proba(x)[:, 1]
    predicted_nodes = infer.get_highest_risk_nodes(baseline_pred, 1)
    infected_nodes = an.get_infected_nodes(t_all,
                                           range(params['runtime'], params['runtime'] + params['prediction_time']))
    comparison_nodes = rng.binomial(n=1, p=0.01, size=g.number_of_nodes())

    for v, a in enumerate(infected_nodes):
        random_scores[i, v] = confusion_matrix(a, predicted_nodes)[1, 1]
        random_comparison_scores[i, v] = confusion_matrix(a, comparison_nodes)[1, 1]
    random_scores[i, -1] = np.sum(infected_nodes)

print(np.average(np.sum(random_scores[:, :-1], 1) / random_scores[:, -1]) * 100)
print(np.average(np.sum(random_comparison_scores[:, :-1], 1) / random_scores[:, -1]) * 100)
# 9.982500726829725
# 0.6150978564771669
