import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tick.base import TimeFunction

import src.utilities as ut


class SimuBaseline:
    def __init__(self, n_nodes, network_type, seed=None, ):
        self.n_nodes = n_nodes
        self.network_type = network_type
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.network = None
        self.adjacency = None
        self.assortativity = None
        self.features = None
        self.node_mu = None
        self.sum_features = None
        self.time_functions = None
        self._set_network()
        self._set_adjacency()

    def _set_network(self):
        if self.network_type == 'barabasi_albert':
            # Create a graph with varying number of edges, similar to social networks
            g = nx.barabasi_albert_graph(self.n_nodes, 1, seed=self.seed)
        elif self.network_type == 'newman_watts_strogatz':
            # Create a graph with more even distribution of edges
            g = nx.newman_watts_strogatz_graph(self.n_nodes, 2, 0.2, seed=self.seed)
        elif self.network_type == 'path':
            g = nx.path_graph(self.n_nodes)
        elif self.network_type == 'star':
            g = nx.star_graph(self.n_nodes)
        else:
            raise ValueError('Unknown network type')
        self.network = g

    def _set_adjacency(self):
        self.adjacency = nx.to_numpy_array(self.network)

    def _homophilic_feature(self, values=None, node_centers=None, max_nodes=None, base=0, ):
        node_list = []
        node_attribute = {i: base for i in self.network.nodes}
        if max_nodes is None:
            max_nodes = self.network.number_of_nodes()
        if node_centers is None:
            node_centers = self.rng.choice(list(self.network.nodes), max_nodes, replace=False)

        for node in node_centers:
            if len(node_list) >= max_nodes:
                break
            if node in node_list:
                continue
            value = node if values is None else self.rng.choice(values)
            node_attribute[node] = value
            node_list.append(node)

            for n in self.network.neighbors(node):
                if len(node_list) >= max_nodes:
                    break
                if n in node_list:
                    continue
                node_attribute[n] = value
                node_list.append(n)
        return np.array(list(node_attribute.values()))

    # TODO split into two functions
    def plot_network(self, show=True, filename=None, params_dict=None, directory='results'):
        fig = plt.figure(figsize=(8, 5))
        gs = fig.add_gridspec(1, 5)
        ax1 = fig.add_subplot(gs[0, 0:-1])
        ax2 = fig.add_subplot(gs[0, 4])

        # plot network
        pos = nx.spiral_layout(self.network)
        node_color = [k for i, k in self.network.nodes.data(-1)]
        nc = nx.draw_networkx_nodes(self.network, pos=pos, node_size=2, node_color=node_color, cmap=plt.cm.jet,
                                    ax=ax1, label=f"Node mu, assortativity: {np.round(self.assortativity[-1], 3)}")
        plt.colorbar(nc, ax=ax1)
        ax1.legend()
        nx.draw_networkx_edges(self.network, pos, alpha=0.1, ax=ax1)
        ax1.set_title(f'Network Diagram')
        ax1.axis('off')

        # plot degree rank
        degree_sequence = sorted([d for n, d in self.network.degree()], reverse=True)
        ax2.plot(degree_sequence, marker='.')
        ax2.set_title('Degree rank plot')
        ax2.set_xlabel(f'Rank')
        ax2.set_ylabel('Degree')
        plt.tight_layout()
        ut.enhance_plot(fig, show, filename, params_dict, directory)

    def simulate(self, proportions, variation, mean_mu, homophilic=False, end_time=1, row=0, omega=1, phi=0, ):
        self._set_features(proportions, homophilic)
        self._set_node_mu(variation, mean_mu, )
        self._set_assortativity(proportions)
        self._set_sinusoidal_time(end_time, row, omega, phi, )

    def _set_features(self, proportions, homophilic=False, ):
        if homophilic:
            self.features = np.zeros(([self.n_nodes, len(proportions)]))
            for i, p in enumerate(proportions):
                self.features[:, i] = self._homophilic_feature(values=[1], max_nodes=int(self.n_nodes * p))
        else:
            self.features = self.rng.binomial(n=1, p=proportions, size=(self.n_nodes, len(proportions)))

    # TODO improve how node mu is calculated and corrects for negative values
    def _set_node_mu(self, variation, mean_mu=None):
        self.sum_features = np.sum(self.features - np.average(self.features), 1)
        self.node_mu = self.sum_features * variation * mean_mu + mean_mu

    def _set_assortativity(self, proportions):
        self.assortativity = np.zeros(len(proportions) + 1)
        for i in range(len(proportions)):
            nx.set_node_attributes(self.network, dict(enumerate(self.features[:, i])), i)
            self.assortativity[i] = nx.attribute_assortativity_coefficient(self.network, i)
        nx.set_node_attributes(self.network, dict(enumerate(self.node_mu)), -1)
        self.assortativity[-1] = nx.attribute_assortativity_coefficient(self.network, -1)

    # TODO Make this optional to improve performance
    def _set_sinusoidal_time(self, end_time, row, omega, phi, num=100):
        t_values = np.linspace(0, end_time, num=num)
        tfs = []
        for n in self.node_mu:
            y_values = n * (1 + row * np.sin(omega * t_values + phi))
            tfs.append(TimeFunction((t_values, y_values)))
        self.time_functions = tfs

    # TODO Improve
    def plot_node_mu(self, show=True, filename=None, params_dict=None, directory='results'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        ax1.hist(self.node_mu)
        ax1.set_title('Histogram of node baseline (mu)')
        ax1.set_xlabel('Node baseline (mu)')
        ax1.set_ylabel('Frequency')

        ax2.scatter(self.sum_features, self.node_mu)
        ax2.set_title('Sum of node features against baseline')
        ax2.set_ylabel('Node baseline (mu)')
        ax2.set_xlabel('Sum of node features')

        plt.tight_layout()
        ut.enhance_plot(fig, show, filename, params_dict, directory)
