import numpy as np
from sklearn.linear_model import LogisticRegression
import mlflow
import time
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels
import src as src

#######################################################################################################################
# INITIALISE

setup = dict(
    start_time=time.time(),
    output_dir=src.set_directory('results/infer_multi_combined/'),
    show_fig=True,
    save_results=False,
    verbose=False,
    percentages=np.array([0.5, 1, 2, 5, 10, 20, 40, 60, 80, 100]),
)

params = dict(
    training_time=150,  # training time, which could be in hours, days or weeks depending on scenario
    prediction_time=50,  # testing time
    average_events=0.2,  # expected number of random events averaged over training time and all nodes
    contagion=0.1,  # expected number of "direct" triggered events
    lifetime=5,  # average time between initial and triggered event
    n_nodes=2000,  # number of nodes
    network_type='barabasi_albert',  # type of graph, either barabasi_albert or newman_watts_strogatz
    seed=0,
    feature_proportions = [0.5,0.5,0.5,0.5,0.5],
    homophilic=False,
    feature_variation=2,  # applies a correlation between node features and node risk
    row=0,  # 0.5,
    omega=1,  # 0.05,
    phi=0,  # np.pi,
)

#######################################################################################################################
# SIMULATE

# Create network and baseline intensity
bs = src.SimuBaseline(n_nodes=params['n_nodes'], network_type=params['network_type'], seed=params['seed'])
bs.set_features(average_events=params['average_events'], proportions=params['feature_proportions'],
                variation=params['feature_variation'], homophilic=params['homophilic'])
bs.set_sinusoidal_time(params['training_time'] + params['prediction_time'],
                       params['row'], params['omega'], params['phi'])
bs.plot_node_average(show=setup['show_fig'], filename='baseline_variation.png', directory=setup['output_dir'])
bs.plot_network(feature=0, show=setup['show_fig'], filename='network.png', directory=setup['output_dir'])

# Generate timestamps
simu = SimuHawkesExpKernels(adjacency=params['contagion'] * bs.adjacency,
                            decays=1 / params['lifetime'],
                            baseline=bs.time_functions,
                            seed=params['seed'],
                            end_time=params['training_time'] + params['prediction_time'],
                            verbose=setup['verbose'],
                            force_simulation=True)
simu.simulate()
t = simu.timestamps
src.plot_timestamps(t, show=setup['show_fig'], filename='timestamps.png', directory=setup['output_dir'])

#######################################################################################################################
# INFER AND PREDICT

infected_nodes = src.get_infected_nodes(t, range(params['training_time'],
                                                 params['training_time'] + params['prediction_time']))

# Contagion Model
model_contagion = src.HawkesExpKernelIdentical(bs.network, verbose=setup['verbose'])
model_contagion.fit(t, training_time=params['training_time'],
                    row=params['row'], omega=params['omega'], phi=params['phi'])
risk_contagion = model_contagion.predict_proba(range(params['training_time'],
                                                     params['training_time'] + params['prediction_time'] - 1))
cm_contagion = src.confusion_matrix(infected_nodes, risk_contagion, setup['percentages'])

# Demographic Model
y_demographic = src.get_infected_nodes(t, (0, params['training_time']))[0]
model_demographic = LogisticRegression(random_state=params['seed'])
model_demographic.fit(bs.features, y_demographic)
risk_demographic = np.array([model_demographic.predict_proba(bs.features)[:, 1]
                             for _ in range(params['prediction_time'] - 1)])
cm_demographic = src.confusion_matrix(infected_nodes, risk_demographic, setup['percentages'])

# Combined Model
risk_combined = risk_contagion[:] / np.sum(risk_contagion, 1)[:, np.newaxis] + \
                risk_demographic[:] / np.sum(risk_demographic, 1)[:, np.newaxis]
cm_combined = src.confusion_matrix(infected_nodes, risk_combined, setup['percentages'])

# Random Model
rng = np.random.default_rng(params['seed'])
risk_random = rng.uniform(0, 1, size=[params['prediction_time'] - 1, params['n_nodes']])
cm_random = src.confusion_matrix(infected_nodes, risk_random, setup['percentages'])

src.plot_cdf({'contagion': cm_contagion, 'demographic': cm_demographic, 'combined': cm_combined, 'random': cm_random},
             setup['percentages'], time=params['prediction_time'] - 1,
             show=setup['show_fig'], filename='cdf.png', directory=setup['output_dir'], )
setup['end_time'] = time.time()

########################################################################################################################
## SAVE

if setup['save_results']:
    # Remote
    # mlflow.set_tracking_uri('databricks')
    # mlflow.set_experiment('/Users/soumaya.mauthoor.17@ucl.ac.uk/contagion_simulation')

    # Locally
    mlflow.set_tracking_uri('./mlruns')
    mlflow.set_experiment('/infer_multi_combined')

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_artifacts(setup['output_dir'])
        mlflow.log_metric('time', setup['end_time'] - setup['start_time'])
