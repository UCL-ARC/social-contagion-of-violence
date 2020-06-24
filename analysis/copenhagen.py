import pandas as pd
import numpy as np
import networkx as nx
import os
import mlflow

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end

from tick.hawkes import SimuHawkesExpKernels
import src.analyse as an
import src.shuffle as sh
import src.homophily as ho
import src.common as co

# Set Parameters
input_dir = 'input/copenhagen/'
output_dir = co.set_directory('analysis/results/copenhagen/')
shuffles = 100
seed = 1

# Create Network
fb_friends = pd.read_csv(input_dir + 'fb_friends.csv')
g = nx.from_pandas_edgelist(fb_friends, 'user_a', 'user_b')
an.plot_network(g, min_edges=50, filename='0_facebook_network.png', directory=output_dir)
nodes_list = list(range(max(g.nodes) + 1))

# Create timestamps
calls = pd.read_csv(input_dir + 'calls.csv')
calls = calls[calls['to'].isin(nodes_list)]
calls = calls[calls['duration'] > 0]
sms = pd.read_csv(input_dir + 'sms.csv')
sms = sms[sms['to'].isin(nodes_list)]

# Analysis
for df_name, df in {'calls': calls, 'sms': sms}.items():
    timestamps = co.group_sort_timestamps(np.array(df['timestamp'] / 1000), np.array(df['to']), nodes_list)

    an.plot_timestamps(timestamps, node_list=g.nodes,
                       filename=f'1_{df_name}_timestamps.png', directory=output_dir)
    ho.set_feature_timestamps(g, timestamps, )
    homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)
    print(f'{df_name} homophily based on event occurence: {homophily}')
    ho.set_feature_timestamps(g, timestamps, use_length=True)
    homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)
    print(f'{df_name} homophily based on event count: {homophily}')

    # Compare with timestamps shuffled along all nodes
    timestamps_shuffled = sh.shuffle_timestamps(timestamps, shuffle_nodes=list(g.nodes), seed=seed)
    an.plot_timestamps(timestamps_shuffled, node_list=g.nodes,
                       filename=f'2_{df_name}_shuffled_timestamps.png', directory=output_dir)
    ho.set_feature_timestamps(g, timestamps_shuffled, )
    homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)
    print(f'{df_name} shuffled homophily based on event occurence: {homophily}')
    ho.set_feature_timestamps(g, timestamps_shuffled, use_length=True)
    homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)
    print(f'{df_name} shuffled homophily based on event count: {homophily}')

    # Compare with simulation
    baseline = np.array([len(t) for t in timestamps])
    end_time = int(np.max(np.concatenate(timestamps)))
    simulation = SimuHawkesExpKernels(adjacency=np.zeros([len(baseline), len(baseline)]),
                                      decays=0,
                                      baseline=baseline / end_time,
                                      seed=seed,
                                      end_time=end_time,
                                      verbose=True, )
    simulation.simulate()
    an.plot_timestamps(simulation.timestamps, node_list=list(g.nodes),
                       filename=f'3_{df_name}_simulated_timestamps.png', directory=output_dir)

    # Shuffle Test
    sh.plot_shuffle_test(g, timestamps=timestamps, shuffles=shuffles,
                         smallest=True, seed=seed, verbose=True,
                         params_dict={'shuffles': shuffles, 'self_excitation': False},
                         filename=f'4_{df_name}_shuffle_test_smallest.png', directory=output_dir)

    sh.plot_shuffle_test(g, timestamps=timestamps, shuffles=shuffles,
                         smallest=False, seed=seed, verbose=True,
                         params_dict={'shuffles': shuffles, 'self_excitation': False},
                         filename=f'5_{df_name}_shuffle_test_all.png', directory=output_dir)

    sh.plot_shuffle_test(g, timestamps=simulation.timestamps, shuffles=shuffles,
                         smallest=True, seed=seed, verbose=True,
                         params_dict={'shuffles': shuffles, 'self_excitation': False},
                         filename=f'6_{df_name}_simulated_shuffle_test_smallest.png', directory=output_dir)

########################################################################################################################
## SAVE

# Remote
# mlflow.set_tracking_uri('databricks')
# mlflow.set_experiment('/Users/soumaya.mauthoor.17@ucl.ac.uk/copenhagen')
#
# Locally
# mlflow.set_tracking_uri('./mlruns')
# mlflow.set_experiment('/simulation')
#
# with mlflow.start_run():
#     mlflow.log_artifacts(output_dir)