import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti
import networkx as nx
matplotlib.use('module://backend_interagg')

##############################################################################
# Parameters

run_time = 300
decays = 7
baseline = 0.001
contagion = 0.05
n_realizations = 5
n_nodes = 100  # 1000 ~ 10 sec, 2000 ~ 50 sec
self_contagion = True

##############################################################################
# Graph

g = nx.barabasi_albert_graph(n_nodes, 2, seed=20)
adjacency = nx.to_numpy_array(g)
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

def estimate_counts(adj):
    nn1 = sum(sum(adj))
    nn2 = sum(sum(np.linalg.matrix_power(adj, 2)))
    nn3 = sum(sum(np.linalg.matrix_power(adj, 3)))
    ratio = contagion / (1 - contagion)
    infections = baseline * run_time * (n_nodes + nn1*ratio + nn2*ratio*ratio + nn3*ratio*ratio*ratio)
    print(f'Estimated base infections = {round(baseline * run_time * n_nodes)}')
    print(f'Estimated infections = {round(infections)}')
    print(f'Estimated infections with self-infection = {round(infections*(1+ratio))}')

estimate_counts(adjacency)

##############################################################################
# Simulate

self_contagion_matrix = np.eye(n_nodes) if self_contagion else np.zeros((n_nodes, n_nodes))
hawkes = SimuHawkesExpKernels(adjacency=contagion * (adjacency + self_contagion_matrix),
                              decays=decays,
                              baseline=baseline * np.ones(n_nodes),
                              # seed=seed,
                              end_time=run_time,
                              verbose=True, )
# hawkes.simulate()
multi = SimuHawkesMulti(hawkes, n_realizations)
multi.simulate()
print(f'Actual infections = {round(np.average(multi.n_total_jumps))}')