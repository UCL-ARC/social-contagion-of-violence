import numpy as np
import matplotlib.pyplot as plt
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
    show_fig=True,
    verbose=False,
    percentages=np.array([0.5, 1, 2, 5, 10, 20, 40, 60, 80, 100]),
    seeds=range(1),
    n_shuffles=1000,
    experiment_name=None,
)

for seed in setup['seeds']:
    for param_name, param in params.items():
        setup['output_dir'] = src.set_directory(os.path.join('results/', param_name), clear=True)
        #######################################################################################################################
        # SIMULATE

        # Create network and baseline intensity
        bs = src.SimuBaseline(n_nodes=param.n_nodes, network_type=param.network_type, seed=seed)
        bs.simulate(mean_mu=param.mean_mu, proportions=param.feature_proportions,
                    variation=param.feature_variation, homophilic=param.homophilic,
                    end_time=param.run_time, row=param.row, omega=param.omega, phi=param.phi)
        bs.plot_node_mu(show=setup['show_fig'], filename='baseline_variation.png', directory=setup['output_dir'])
        bs.plot_network(show=setup['show_fig'], filename='network.png', directory=setup['output_dir'])

        # Generate timestamps
        simu = SimuHawkesExpKernels(adjacency=param.alpha * bs.adjacency, baseline=bs.time_functions,
                                    decays=1 / param.lifetime, end_time=param.run_time,
                                    seed=seed, verbose=setup['verbose'], force_simulation=False)
        simu.threshold_negative_intensity(True)
        simu.simulate()
        t = simu.timestamps
        _, event_ast = src.plot_timestamps(t, bs.network, simulation=simu, show=setup['show_fig'],
                                           filename='timestamps.png', directory=setup['output_dir'])
        _, shuffle = src.plot_shuffle_test(bs.network, t, setup['n_shuffles'], seed=seed, verbose=setup['verbose'],
                                           show=setup['show_fig'], filename='shuffle_test.png',
                                           directory=setup['output_dir'])

        #######################################################################################################################
        # INFER AND PREDICT

        infected_nodes = src.get_infected_nodes(t, range(param.training_time, param.run_time))

        # Contagion Model
        model_contagion = src.HawkesExpKernelIdentical(bs.network, verbose=setup['verbose'])
        model_contagion.fit(t, training_time=param.training_time, row=param.row, omega=param.omega, phi=param.phi)
        risk_contagion = model_contagion.predict_proba(range(param.training_time, param.run_time - 1))
        cm_contagion = src.confusion_matrix(infected_nodes, risk_contagion, setup['percentages'])

        # logistic Model
        y_logistic = src.get_infected_nodes(t, (0, param.training_time))[0]
        model_logistic = LogisticRegression(random_state=seed)
        model_logistic.fit(bs.features, y_logistic)
        risk_logistic = np.array([model_logistic.predict_proba(bs.features)[:, 1]
                                  for _ in range(param.prediction_time - 1)])
        cm_logistic = src.confusion_matrix(infected_nodes, risk_logistic, setup['percentages'])

        # Combined Model
        risk_combined = src.norm(risk_contagion) + src.norm(risk_logistic)
        cm_combined = src.confusion_matrix(infected_nodes, risk_combined, setup['percentages'])

        # Random Model
        rng = np.random.default_rng(seed)
        risk_random = rng.uniform(0, 1, size=[param.prediction_time - 1, param.n_nodes])
        cm_random = src.confusion_matrix(infected_nodes, risk_random, setup['percentages'])

        src.plot_cdf(
            {f'Hawkes ({model_contagion.label})': cm_contagion,
             f"Logistic ($\\beta_L$:{src.round_to_n(np.average(model_logistic.coef_[0]), 3)})": cm_logistic,
             'Combined': cm_combined, 'Random': cm_random},
            setup['percentages'], time=param.prediction_time - 1,
            show=setup['show_fig'], filename='cdf.png', directory=setup['output_dir'], )

        ########################################################################################################################
        ## SAVE

        if setup['experiment_name'] is not None:
            # Remote
            # mlflow.set_tracking_uri('databricks')
            # mlflow.set_experiment('/Users/soumaya.mauthoor.17@ucl.ac.uk/contagion/'+ param.experiment_name'])

            # Locally
            mlflow.set_tracking_uri('./mlruns')
            mlflow.set_experiment(setup['experiment_name'])

            with mlflow.start_run(run_name=param_name):
                mlflow.log_param('seed', seed)
                mlflow.log_params(asdict(param))
                mlflow.log_metric('events_training', model_contagion.n_training_jumps)
                mlflow.log_metric('events_testing', simu.n_total_jumps - model_contagion.n_training_jumps)
                mlflow.log_metric('ast_mu', bs.assortativity[-1])
                mlflow.log_metric('ast_occurrence', event_ast[0])
                mlflow.log_metric('ast_count', event_ast[1])
                mlflow.log_metric('shuffle_diff', shuffle[0])
                mlflow.log_metric('shuffle_zscore', shuffle[1])
                mlflow.log_metric('hawkes_mu', model_contagion.mu_est)
                mlflow.log_metric('hawkes_alpha', model_contagion.alpha_est)
                mlflow.log_metric('hawkes_beta', model_contagion.beta_est)
                mlflow.log_metric('logistic_coeff', np.average(model_logistic.coef_[0]))
                mlflow.log_metric('HR1_hawkes', cm_contagion[1][3] / np.sum(cm_contagion[1][2:3]))
                mlflow.log_metric('HR1_logistic', cm_logistic[1][3] / np.sum(cm_logistic[1][2:3]))
                mlflow.log_metric('HR1_combined', cm_combined[1][3] / np.sum(cm_combined[1][2:3]))
                mlflow.log_artifacts(setup['output_dir'])

        print(f'{seed}: {param_name}, {model_contagion.coef_}')
        plt.close('all')
