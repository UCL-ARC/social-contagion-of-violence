import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels

import simulate_infections as si
import shuffle as st

directory = si.set_results_directory('copenhagen')

# Create Network
fb_friends = pd.read_csv('data/copenhagen/fb_friends.csv')
g = nx.from_pandas_edgelist(fb_friends, 'user_a', 'user_b')
adj_array = nx.to_numpy_array(g)
# si.plot_network(g, min_edges=50, filename='facebook_network.png', directory=directory)
nodes_list = pd.unique(fb_friends[['user_a', 'user_b']].values.ravel())
nodes_list.sort()

# Create timestamps
calls = pd.read_csv('data/copenhagen/calls.csv')
calls = calls[calls['to'].isin(nodes_list)]
sms = pd.read_csv('data/copenhagen/sms.csv')
sms = sms[sms['to'].isin(nodes_list)]


for df_name, df in {'calls':calls,'sms':sms}.items():
    timestamps = st.group_timestamps_by_node(np.array(df['timestamp']), np.array(df['to']), nodes_list)

    si.set_homophily(timestamps, g, node_list=nodes_list)
    homophily = np.round(nx.numeric_assortativity_coefficient(g, 'feature'), 3)
    print(f'{df_name} homophily: {homophily}')

    timestamps_diffs = st.diff_neighbours(adj_array, timestamps)
    timestamps_shuffled_diffs = st.repeat_shuffle_diff(adj=adj_array, timestamps=timestamps,
                                                       repetitions=100, seed=1, verbose=True)
    st.plot_diff_comparison(timestamps_diffs, timestamps_shuffled_diffs,
                            filename=f'{df_name}_timestamps_diff.png', directory=directory)

# n_nodes = 1000
# random_simu = SimuHawkesExpKernels(adjacency=np.zeros([n_nodes,n_nodes]),
#                                       decays=0,
#                                       baseline=len(calls) * np.ones(n_nodes) / (max(calls['timestamp'])*n_nodes),
#                                       seed=1,
#                                       end_time=max(calls['timestamp']),
#                                       verbose=True, )
# random_simu.simulate()

