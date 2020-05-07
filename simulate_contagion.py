import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti
import networkx as nx
matplotlib.use('module://backend_interagg')

##############################################################################
# Parameters

run_time = 1000
decays = 7
baseline = 0.0005
contagion = 0.000
n_realizations = 20
n_nodes = 1000  # 1000 ~ 10 sec, 2000 ~ 50 sec
self_contagion = False

##############################################################################
# Graph

g = nx.barabasi_albert_graph(n_nodes, 2, seed=20)
adjacency = nx.to_numpy_array(g)

# adjacency = np.eye(n_nodes,k=1) + np.eye(n_nodes,k=-1)
# g = nx.from_numpy_array(adjacency)

plt.hist(np.average(adjacency,0)*n_nodes,100)
plt.show()

# For small graphs:
def plot_graph(g1):
    if len(g1.nodes) <= 100:
        pos = nx.spring_layout(g1)
        nx.draw(g1, pos=pos)
        nx.draw_networkx_labels(g1, pos=pos)
        plt.show()
    else:
        print('graph is too large to plot')

plot_graph(g)

##############################################################################
# Estimates

def estimate_counts(adj, baseline, run_time, contagion, nn=3):
    ratio = contagion / (1 - contagion)

    def nn_contribution(nn):
        nnx = [sum(sum(np.linalg.matrix_power(adj, i))) * np.power(ratio, i) for i in range(1,nn+1)]
        return sum(nnx)

    infections = baseline * run_time * (n_nodes + nn_contribution(nn))
    print(f'Estimated base infections = {round(baseline * run_time * n_nodes)}')
    print(f'Estimated infections = {round(infections)}')
    print(f'Estimated infections with self-infection = {round(infections * (1 + ratio))}')

estimate_counts(adjacency, baseline, run_time, contagion, nn=20)

##############################################################################
# Simulate contagion

self_contagion_matrix = np.eye(n_nodes) if self_contagion else np.zeros((n_nodes, n_nodes))
all_timestamps = []
n_total_jumps = []
for i in range (n_realizations):
    hawkes = SimuHawkesExpKernels(adjacency=contagion * (adjacency + self_contagion_matrix),
                                  decays=decays,
                                  baseline=baseline * np.ones(n_nodes),
                                  # seed=seed,
                                  end_time=run_time,
                                  verbose=True, )
    hawkes.simulate()
    n_total_jumps.append(hawkes.n_total_jumps)
    all_timestamps.append(hawkes.timestamps)

print(f'Actual average infections = {round(np.average(n_total_jumps))}')
print(f'Actual std infections = {round(np.std(n_total_jumps))}')
plt.hist(n_total_jumps)
plt.show()

# There is a multi-threaded solution but it's slower:
# multi = SimuHawkesMulti(hawkes, n_realizations, n_threads=0)
# multi.simulate()
