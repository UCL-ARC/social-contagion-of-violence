import numpy as np
import networkx as nx
from scipy.optimize import minimize, basinhopping, brute
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import mlflow
import time
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels
import src as src

#######################################################################################################################
# INITIALISE

start_time = time.time()
output_dir = src.set_directory('results/infer_multi_combined/')
show_fig = True
save_results = False

params = dict(
    training_time=150,  # training time, which could be in hours, days or weeks depending on scenario
    prediction_time=10,  # testing time
    average_events=0.3,  # expected number of events averaged over training time and all nodes
    contagion=0.1,  # expected number of "direct" triggered events
    lifetime=5,  # average time between initial and triggered event
    n_nodes=1000,  # number of nodes
    network_type='barabasi_albert',  # type of graph, either barabasi_albert or newman_watts_strogatz
    n_realizations=10,  # how many times to repeat simulation and inference
    seed=1,
    feature_description={'gender': {'values': (-1, 1), 'p': (0.9, 0.1), 'homophilic': False},
                         'age': {'values': range(-2, 2), 'p': (0.4, 0.3, 0.2, 0.1), },
                         'arrests': {'values': range(-2, 2), 'p': (0.4, 0.3, 0.2, 0.1)}},
    feature_variation=1, # how much the features vary by, at 0 all the nodes have the same feature value
)

#######################################################################################################################
# SIMULATE

# Create network and baseline intensity
bs = src.SimuBaseline(n_nodes=params['n_nodes'], network_type=params['network_type'], seed=params['seed'])
bs.set_features(average_events=params['average_events'], feature_description=params['feature_description'],
                feature_variation=params['feature_variation'])
bs.plot_node_average(show=show_fig, filename='baseline_variation.png', directory=output_dir)
bs.plot_network(show=show_fig, feature='gender', filename='network.png', directory=output_dir)

# Generate timestamps
simu = SimuHawkesExpKernels(adjacency=params['contagion'] * bs.adjacency,
                            decays=1 / params['lifetime'],
                            baseline=bs.node_average / params['training_time'],
                            seed=params['seed'],
                            end_time=params['training_time'] + params['prediction_time'],
                            verbose=False)
timestamps = src.repeat_simulations(simu, params['n_realizations'])
print([len(np.concatenate(i)) for i in timestamps])
# [446, 452, 454, 485, 434, 401, 549, 471, 591, 488]
src.plot_multi_timestamps(timestamps, show=show_fig, filename='multi_timestamps.png', directory=output_dir)

#######################################################################################################################
# INFER AND PREDICT

coeffs = np.zeros([params['n_realizations'], 7])
counts_demographic = np.zeros([params['n_realizations'], params['prediction_time']])
counts_contagion = np.zeros([params['n_realizations'], params['prediction_time']])
counts_combined = np.zeros([params['n_realizations'], params['prediction_time']])
counts_comparison = np.zeros([params['n_realizations'], params['prediction_time']])
counts_infection = np.zeros([params['n_realizations'], params['prediction_time']])

# TODO calculate range based on data
contagion_coeffs_range = (slice(1e-10, 0.1 / params['training_time'], 0.02 / params['training_time']),  # mu range
                          slice(1e-10, 0.2, 0.04),  # alpha range
                          slice(1e-10, 0.5, 0.1))  # beta range

for i, t_all in enumerate(timestamps):
    t, t_right = src.split_timestamps(t_all, params['training_time'])

    # TODO Turn this into model with .fit() method
    # Infer Hawkes Parameters
    brute_res = brute(src.crit_multi, ranges=contagion_coeffs_range, full_output=True, finish=0,
                      args=(bs.network, t, params['training_time'], 0, 1, 0, False), )
    ggd_res = minimize(src.crit_multi, x0=brute_res[0],
                       args=(bs.network, t, params['training_time'], 0, 1, 0, False),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, 1), (1e-10, None)))
    print(ggd_res.x)
    coeffs[i, 0:3] = ggd_res.x

    # Infer Demographic Parameters
    src.set_feature_timestamps(bs.network, t)
    y = np.array(list(nx.get_node_attributes(bs.network, 'feature').values()))
    model = LogisticRegression(random_state=params['seed'])
    model.fit(bs.features, y)
    coeffs[i, 3:7] = np.concatenate([model.coef_[0], model.intercept_])

    # Predict highest risk nodes
    contagion_risk = src.contagion_risk(bs.network, t_all, *ggd_res.x[1:],
                                        range(params['training_time'],
                                              params['training_time'] + params['prediction_time']))
    baseline_risk = model.predict_proba(bs.features)[:, 1]
    predicted_demographic = np.array([src.get_highest_risk_nodes(baseline_risk)
                                      for day in range(params['prediction_time'])])
    predicted_contagion = np.array([src.get_highest_risk_nodes(day) for day in contagion_risk])
    # predicted_combined = np.array([src.get_highest_risk_nodes(day / sum(day) + baseline_risk / sum(baseline_risk))
    #                                for day in contagion_risk])
    predicted_combined = np.zeros([params['prediction_time'], params['n_nodes']])
    predicted_combined[np.logical_or(predicted_contagion == True, predicted_demographic == True)] = 1

    # Obtain infected and comparison nodes
    infected_nodes = src.get_infected_nodes(t_all,
                                            range(params['training_time'],
                                                  params['training_time'] + params['prediction_time']))
    rng = np.random.default_rng(i)
    comparison_nodes = rng.binomial(n=1, p=0.01, size=[params['prediction_time'], params['n_nodes']])

    # TODO Store other counts as well
    # Count predictions
    for day in range(params['prediction_time']):
        counts_demographic[i, day] = confusion_matrix(infected_nodes[day], predicted_demographic[day])[1, 1]
        counts_contagion[i, day] = confusion_matrix(infected_nodes[day], predicted_contagion[day])[1, 1]
        counts_combined[i, day] = confusion_matrix(infected_nodes[day], predicted_combined[day])[1, 1]
        counts_comparison[i, day] = confusion_matrix(infected_nodes[day], comparison_nodes[day])[1, 1]
        counts_infection[i, day] = np.sum(infected_nodes[day])

end_time = time.time()

########################################################################################################################
## ANALYSE

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
# 0.14417394681006296
# 0.014837662337662339

########################################################################################################################
## SAVE

if save_results:
    pickle.dump(timestamps, open(output_dir + 'counts_timestamps.p', 'wb'))
    pickle.dump(coeffs, open(output_dir + 'coeffs.p', 'wb'))
    pickle.dump(counts_demographic, open(output_dir + 'counts_demographic.p', 'wb'))
    pickle.dump(counts_contagion, open(output_dir + 'counts_contagion.p', 'wb'))
    pickle.dump(counts_combined, open(output_dir + 'counts_combined.p', 'wb'))
    pickle.dump(counts_comparison, open(output_dir + 'counts_comparison.p', 'wb'))
    pickle.dump(counts_infection, open(output_dir + 'counts_infection.p', 'wb'))

    # Remote
    # mlflow.set_tracking_uri('databricks')
    # mlflow.set_experiment('/Users/soumaya.mauthoor.17@ucl.ac.uk/contagion_simulation')
    #
    # Locally
    mlflow.set_tracking_uri('./mlruns')
    mlflow.set_experiment('/infer_multi_combined')

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_artifacts(output_dir)
        mlflow.log_metric('time', end_time - start_time)
