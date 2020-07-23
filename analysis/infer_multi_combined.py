import numpy as np
from sklearn.linear_model import LogisticRegression
from dataclasses import asdict
import mlflow
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels
import src as src
from input.parameters import params

#######################################################################################################################
# INITIALISE

setup = dict(
    output_dir=src.set_directory('results/infer_multi_combined/'),
    show_fig=True,
    save_results=False,
    verbose=False,
    percentages=np.array([0.5, 1, 2, 5, 10, 20, 40, 60, 80, 100]),
    n_realizations=1,
    n_shuffles=1000,
    experiment_name='29SEP',
)

for seed in range(setup['n_realizations']):
    for param_name, param in params.items():
        #######################################################################################################################
        # SIMULATE

        # Create network and baseline intensity
        bs = src.SimuBaseline(n_nodes=param.n_nodes, network_type=param.network_type, seed=seed)
        bs.set_features(average_events=param.average_events, proportions=param.feature_proportions,
                        variation=param.feature_variation, homophilic=param.homophilic)
        bs.set_sinusoidal_time(param.run_time, param.row, param.omega, param.phi)
        bs.plot_node_average(show=setup['show_fig'], filename='baseline_variation.png', directory=setup['output_dir'])
        bs.plot_network(feature=0, show=setup['show_fig'], filename='network.png', directory=setup['output_dir'])

        # Generate timestamps
        simu = SimuHawkesExpKernels(adjacency=param.contagion * bs.adjacency,
                                    decays=1 / param.lifetime, baseline=bs.time_functions,
                                    end_time=param.run_time, seed=seed,
                                    verbose=setup['verbose'], force_simulation=True)
        simu.track_intensity(intensity_track_step=param.run_time)
        simu.simulate()
        t = simu.timestamps
        src.plot_timestamps(t, show=setup['show_fig'], filename='timestamps.png', directory=setup['output_dir'])

        #######################################################################################################################
        # INFER AND PREDICT

        infected_nodes = src.get_infected_nodes(t, range(param.training_time, param.run_time))

        # Contagion Model
        model_contagion = src.HawkesExpKernelIdentical(bs.network, verbose=setup['verbose'])
        model_contagion.fit(t, training_time=param.training_time, row=param.row, omega=param.omega, phi=param.phi)
        risk_contagion = model_contagion.predict_proba(range(param.training_time, param.run_time - 1))
        cm_contagion = src.confusion_matrix(infected_nodes, risk_contagion, setup['percentages'])

        # Demographic Model
        y_demographic = src.get_infected_nodes(t, (0, param.training_time))[0]
        model_demographic = LogisticRegression(random_state=seed)
        model_demographic.fit(bs.features, y_demographic)
        risk_demographic = np.array([model_demographic.predict_proba(bs.features)[:, 1]
                                     for _ in range(param.prediction_time - 1)])
        cm_demographic = src.confusion_matrix(infected_nodes, risk_demographic, setup['percentages'])

        # Combined Model
        risk_combined = src.norm(risk_contagion) + src.norm(risk_demographic)
        cm_combined = src.confusion_matrix(infected_nodes, risk_combined, setup['percentages'])

        # Random Model
        rng = np.random.default_rng(seed)
        risk_random = rng.uniform(0, 1, size=[param.prediction_time - 1, param.n_nodes])
        cm_random = src.confusion_matrix(infected_nodes, risk_random, setup['percentages'])

        src.plot_cdf(
            {'contagion': cm_contagion, 'demographic': cm_demographic, 'combined': cm_combined, 'random': cm_random},
            setup['percentages'], time=param.prediction_time - 1,
            show=setup['show_fig'], filename='cdf.png', directory=setup['output_dir'], )

        ########################################################################################################################
        ## SAVE

        if setup['save_results']:
            # Remote
            # mlflow.set_tracking_uri('databricks')
            # mlflow.set_experiment('/Users/soumaya.mauthoor.17@ucl.ac.uk/contagion/'+ param.experiment_name'])

            # Locally
            mlflow.set_tracking_uri('./mlruns')
            mlflow.set_experiment(setup['experiment_name'])

            with mlflow.start_run(run_name=param_name):
                mlflow.set_tag('seed', seed)
                mlflow.log_params(asdict(param))
                mlflow.log_metric('n_training_jumps', model_contagion.n_training_jumps)
                mlflow.log_metric('n_testing_jumps', simu.n_total_jumps - model_contagion.n_training_jumps)
                mlflow.log_metric('contagion_mu', model_contagion.mu)
                mlflow.log_metric('contagion_alpha', model_contagion.alpha)
                mlflow.log_metric('contagion_beta', model_contagion.beta)
                mlflow.log_metrics({f'demographic_{i}': v for i, v in enumerate(model_demographic.coef_[0])})
                mlflow.log_artifacts(setup['output_dir'])

        print(f'{seed}: {param_name}')
