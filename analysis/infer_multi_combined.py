import numpy as np
from scipy.optimize import minimize, basinhopping, brute
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
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
    runtime=150,  # training time in days
    prediction_time=10,  # testing time in days
    baseline=0.3,  # expected number of times node becomes randomly infected over runtime
    contagion=0.1,  # expected number of "direct" triggered infections
    self_contagion=False,  # whether people can reinfect themselves
    total_rate=0.5,  # expected total number of infections per node over runtime
    lifetime=5,  # average time between initial and triggered infection in days
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
# SET BASELINE

# ho.set_feature_clustered(g, values=[2], base=1, max_nodes=int(params['n_nodes'] * 0.5), seed=params['seed'])
# print(nx.numeric_assortativity_coefficient(g, 'feature'))

rng = np.random.default_rng(params['seed'])
gender = rng.choice((-1, 1), size=params['n_nodes'], p=(0.9, 0.1))
age = rng.choice(range(-2, 2), size=params['n_nodes'], p=(0.4, 0.3, 0.2, 0.1))
arrests = rng.choice(range(-2, 2), size=params['n_nodes'], p=(0.4, 0.3, 0.2, 0.1))
demographic = gender + age + arrests
demographic = 1 / (1 + np.exp(-demographic))
demographic = params['baseline'] * demographic * params['n_nodes'] / sum(demographic)
x = np.array([gender, age, arrests]).T

#######################################################################################################################
# SIMULATE

simu = SimuHawkesExpKernels(adjacency=params['contagion'] * adj_array,
                            decays=1 / params['lifetime'],
                            baseline=demographic / params['runtime'],
                            seed=params['seed'],
                            end_time=params['runtime'] + params['prediction_time'],
                            verbose=True)
timestamps = an.repeat_simulations(simu, params['n_realizations'])
print([len(np.concatenate(i)) for i in timestamps])
# [446, 452, 454, 485, 434, 401, 549, 471, 591, 488]
an.plot_multi_timestamps(timestamps)

#######################################################################################################################
# INFER AND PREDICT

coeffs = np.zeros([params['n_realizations'], 7])
counts_demographic = np.zeros([params['n_realizations'], params['prediction_time']])
counts_contagion = np.zeros([params['n_realizations'], params['prediction_time']])
counts_combined = np.zeros([params['n_realizations'], params['prediction_time']])
counts_comparison = np.zeros([params['n_realizations'], params['prediction_time']])
counts_infection = np.zeros([params['n_realizations'], params['prediction_time']])

for i, t_all in enumerate(timestamps):
    t, t_right = an.split_timestamps(t_all, params['runtime'])

    # Infer Hawkes Parameters
    brute_res = brute(infer.crit_multi, ranges=coeffs_range, args=(g, t, params['runtime'], 0, 1, 0, False),
                      full_output=True, finish=0, )
    ggd_res = minimize(infer.crit_multi, x0=brute_res[0], args=(g, t, params['runtime'], 0, 1, 0, False),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, 1), (1e-10, None)))
    print(ggd_res.x)
    coeffs[i, 0:3] = ggd_res.x

    # Infer Demographic Parameters
    ho.set_feature_timestamps(g, t)
    y = np.array(list(nx.get_node_attributes(g, 'feature').values()))
    model = LogisticRegression(random_state=params['seed'])
    model.fit(x, y)
    coeffs[i, 3:7] = np.concatenate([model.coef_[0], model.intercept_])

    # Predict highest risk nodes
    contagion_risk = infer.contagion_risk(g, t_all, *ggd_res.x[1:],
                                          range(params['runtime'], params['runtime'] + params['prediction_time']))
    baseline_risk = model.predict_proba(x)[:, 1]
    predicted_demographic = infer.get_highest_risk_nodes(baseline_risk)
    predicted_contagion = np.array([infer.get_highest_risk_nodes(day) for day in contagion_risk])
    predicted_combined = np.array([infer.get_highest_risk_nodes(day * baseline_risk) for day in contagion_risk])

    # Obtain infected and comparison nodes
    infected_nodes = an.get_infected_nodes(t_all,
                                           range(params['runtime'], params['runtime'] + params['prediction_time']))
    rng = np.random.default_rng(i)
    comparison_nodes = rng.binomial(n=1, p=0.01, size=[params['prediction_time'], g.number_of_nodes()])

    # Count predictions
    for day in range(params['prediction_time']):
        counts_demographic[i, day] = confusion_matrix(infected_nodes[day],predicted_demographic)[1, 1]
        counts_contagion[i, day] = confusion_matrix(infected_nodes[day],predicted_contagion[day])[1, 1]
        counts_combined[i, day] = confusion_matrix(infected_nodes[day],predicted_combined[day])[1, 1]
        counts_comparison[i, day] = confusion_matrix(infected_nodes[day],comparison_nodes[day])[1, 1]
        counts_infection[i, day] = np.sum(infected_nodes[day])

print(np.average(coeffs, 0))
print(np.std(coeffs, 0))
# [0.00204025 0.09917007 0.19875739 0.67594209 0.7302766  0.67913671  0.90151766]
# [1.26679547e-04 1.33445879e-02 3.56164654e-02 9.97729627e-02  9.52277820e-02 8.61288939e-02 1.30388188e-01]

print(np.average(np.sum(counts_demographic, 1) / np.sum(counts_infection, 1)))
print(np.average(np.sum(counts_contagion, 1) / np.sum(counts_infection, 1)))
print(np.average(np.sum(counts_combined, 1) / np.sum(counts_infection, 1)))
print(np.average(np.sum(counts_comparison, 1) / np.sum(counts_infection, 1)))
# 0.07059616939154144
# 0.07357777741852152
# 0.04879717647412747
# 0.014837662337662339