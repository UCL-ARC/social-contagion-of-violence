import pandas as pd
import numpy as np
import networkx as nx
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end

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
an.plot_network(g, min_edges=50, filename='facebook_network.png', directory=output_dir)
nodes_list = pd.unique(fb_friends[['user_a', 'user_b']].values.ravel())
nodes_list.sort()

# Create timestamps
calls = pd.read_csv(input_dir + 'calls.csv')
calls = calls[calls['to'].isin(nodes_list)]
calls = calls[calls['duration']>0]
sms = pd.read_csv(input_dir + 'sms.csv')
sms = sms[sms['to'].isin(nodes_list)]

# Analysis
for df_name, df in {'calls': calls}.items():

    raw_timestamps = co.group_sort_timestamps(np.array(df['timestamp'] / 1000), np.array(df['to']), nodes_list)

    min_diff = 1
    timestamps = []
    for node, v in enumerate(raw_timestamps):
        timestamps.append([])
        if len(v) > 0:
            timestamps[node].append(v[0])
            for i in range(len(v) - 1):
                if v[i + 1] - v[i] > min_diff:
                    timestamps[node].append(v[i + 1])

    an.plot_timestamps(timestamps, filename=f'{df_name}_timestamps.png', directory=output_dir)
    ho.set_homophily(timestamps, g, node_list=nodes_list)
    homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)
    print(f'{df_name} homophily based on event occurence: {homophily}')
    ho.set_homophily(timestamps, g, node_list=nodes_list, use_length=True)
    homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)
    print(f'{df_name} homophily based on event count: {homophily}')

    # Compare with timestamps shuffled along all nodes
    timestamps_shuffled = sh.shuffle_timestamps(timestamps, shuffle_nodes=True, seed=seed)
    an.plot_timestamps(timestamps_shuffled, filename=f'{df_name}_shuffled_timestamps.png', directory=output_dir)
    ho.set_homophily(timestamps_shuffled, g, node_list=nodes_list)
    homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)
    print(f'{df_name} shuffled homophily based on event occurence: {homophily}')
    ho.set_homophily(timestamps_shuffled, g, node_list=nodes_list, use_length=True)
    homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)
    print(f'{df_name} shuffled homophily based on event count: {homophily}')

    # Shuffle Test
    sh.plot_shuffle_test(adj=nx.to_numpy_array(g), timestamps=timestamps, shuffles=shuffles,
                         smallest=True, seed=seed, verbose=True,
                         params_dict={'shuffles': shuffles, 'self_excitation': False},
                         filename=f'{df_name}_timestamps_diff.png', directory=output_dir)

    # Shuffle Test with Self-excitation
    sh.plot_shuffle_test(adj=nx.to_numpy_array(g) + np.eye(g.number_of_nodes()),
                         timestamps=timestamps, shuffles=shuffles, smallest=True,
                         seed=seed, verbose=True,
                         params_dict={'shuffles': shuffles, 'self_excitation': True},
                         filename=f'{df_name}_timestamps_diff_self_excitation.png', directory=output_dir)

import matplotlib.pyplot as plt
plt.hist(np.concatenate(timestamps), bins=100, range=(0,500)); plt.show();