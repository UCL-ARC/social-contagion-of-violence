import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tick.hawkes import SimuHawkesExpKernels
from tick.plot import plot_point_process
import networkx as nx
matplotlib.use('module://backend_interagg')

# Simulate network
n_nodes = 1000
# g = nx.random_regular_graph(3,n_nodes)
g = nx.barabasi_albert_graph(n_nodes,2)

# Draw network (for n_nodes <=100):
pos = nx.spring_layout(g)
nx.draw(g,pos=pos)
nx.draw_networkx_labels(g,pos=pos)
plt.show()

# Convert network to matrix
adjacency = nx.to_numpy_array(g)
plt.hist(np.average(adjacency,0)*n_nodes)
plt.show()

# Simulate Hawkes process
run_time = 100
contagion = 0.05
decays = 3 * np.ones((n_nodes, n_nodes))
baseline = 0.01 * np.ones(n_nodes)
hawkes = SimuHawkesExpKernels(adjacency=adjacency*contagion, decays=decays,
                              baseline=baseline, verbose=False, end_time=run_time)
dt = 0.1
hawkes.track_intensity(0.1)
hawkes.simulate()

# Plot Hawkes process (For n_nodes<10):
fig, ax = plt.subplots(n_nodes,1, figsize = (8,32))
plot_point_process(hawkes, n_points=50000, t_min=10, max_jumps=30, ax=ax)
fig.tight_layout
plt.show()

# Plot intensity of a single node compared with timestamps and those of neighbors
node = 2
fig, ax = plt.subplots(1,1,)
ax.plot(hawkes.intensity_tracked_times,hawkes.tracked_intensity[node])
for n in dict(g[node]).keys():
    for t in hawkes.timestamps[n]:
        ax.axvline(x=t,color='black')
for t in hawkes.timestamps[node]:
        ax.axvline(x=t,color='red')
plt.show()

# Calculate neighbours upto cutoff jumps away
nx.single_source_shortest_path(g,0,cutoff=2)

# Calculate distance between events/timestamps of neighbors
dif = []
for start_node, start_timestamps in enumerate(hawkes.timestamps):
    for start_timestamp in start_timestamps:
        for end_node, end_timestamps in enumerate(hawkes.timestamps):
            for end_timestamp in end_timestamps:
                if end_node in g.adj[start_node]:
                    if start_timestamp - end_timestamp > 0:
                        dif.append(start_timestamp -end_timestamp)
plt.hist(dif,bins=50)
plt.show()