import pandas as pd
import numpy as np
import networkx as nx
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels

import src.analyse as an
import src.shuffle as sh
import src.homophily as ho
import src.common as co

input_dir = 'input/copenhagen/'
output_dir = co.set_directory('analysis/results/copenhagen/')

# Create Network
fb_friends = pd.read_csv(input_dir + 'fb_friends.csv')
g = nx.from_pandas_edgelist(fb_friends, 'user_a', 'user_b')
an.plot_network(g, min_edges=50, filename='facebook_network.png', directory=output_dir)
nodes_list = pd.unique(fb_friends[['user_a', 'user_b']].values.ravel())
nodes_list.sort()

# Create timestamps
calls = pd.read_csv(input_dir + 'calls.csv')
calls = calls[calls['to'].isin(nodes_list)]
sms = pd.read_csv(input_dir + 'sms.csv')
sms = sms[sms['to'].isin(nodes_list)]

for df_name, df in {'calls': calls, 'sms': sms}.items():
    timestamps = co.group_sort_timestamps(np.array(df['timestamp'] / 1000), np.array(df['to']), nodes_list)
    an.plot_timestamps(timestamps, filename=f'{df_name}_timestamps.png', directory=output_dir)

    # Simulation
    n_nodes = len(nodes_list)
    end_time = max(df['timestamp']) / 1000
    comparison = SimuHawkesExpKernels(adjacency=np.zeros([n_nodes, n_nodes]),
                                      baseline=np.ones([n_nodes]) * len(df) / (end_time * n_nodes),
                                      end_time=end_time, decays=0, seed=1)
    comparison.simulate()
    an.plot_timestamps(comparison.timestamps, filename=f'{df_name}_simulated_timestamps.png', directory=output_dir)

    # Calculate Homophily
    ho.set_homophily(timestamps, g, node_list=nodes_list)
    homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)
    print(f'{df_name} homophily: {homophily}')

    # Shuffle Test
    shuffles = 100
    adj_array = nx.to_numpy_array(g)
    sh.plot_shuffle_test(adj=adj_array, timestamps=timestamps, shuffles=shuffles, seed=1, verbose=True,
                            params_dict={'shuffles': shuffles, 'self_excitation': True},
                            filename=f'{df_name}_timestamps_diff.png', directory=output_dir)

    # Shuffle Test with Self-excitation
    adj_array = nx.to_numpy_array(g) + np.eye(g.number_of_nodes())
    sh.plot_shuffle_test(adj=adj_array, timestamps=timestamps, shuffles=shuffles, seed=1, verbose=True,
                            params_dict={'shuffles': shuffles, 'self_excitation': True},
                            filename=f'{df_name}_timestamps_diff_self_excitation.png', directory=output_dir)
