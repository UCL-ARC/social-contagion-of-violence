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
        self.node_average = None
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

    def _homophilic_feature(self, values=None, node_centers=None, max_nodes=None, base=-1, ):
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
    def plot_network(self, min_edges=3, feature=None, show=True, filename=None, params_dict=None, directory='results'):
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(1, 5)
        ax1 = fig.add_subplot(gs[0, 0:-1])
        ax2 = fig.add_subplot(gs[0, 4])

        # plot network
        pos = nx.spring_layout(self.network)
        if feature is not None:
            node_color = [k for i, k in self.network.nodes.data(feature)]
            nc = nx.draw_networkx_nodes(self.network, pos, node_size=20, alpha=0.4,
                                        node_color=node_color, ax=ax1, cmap=plt.cm.jet)
            plt.colorbar(nc, ax=ax1)
            ax1.plot([], [], ' ', label=f"Node color: {feature}")
            ax1.plot([], [], ' ', label=f"Assortativity: {np.round(self.assortativity[feature], 3)}")
            ax1.legend()
        else:
            nx.draw_networkx_nodes(self.network, pos, node_size=10, alpha=0.4, ax=ax1, )
        nx.draw_networkx_edges(self.network, pos, alpha=0.4, ax=ax1)
        labels = {n: n for n, d in self.network.degree if d > min_edges}
        nx.draw_networkx_labels(self.network, labels=labels, pos=pos, ax=ax1, font_size=10)
        ax1.set_title(f'Network Diagram')
        ax1.axis('off')

        # plot degree rank
        degree_sequence = sorted([d for n, d in self.network.degree()], reverse=True)
        ax2.plot(degree_sequence, marker='.')
        ax2.set_title('Degree rank plot')
        ax2.set_xlabel(f'rank')
        ax2.set_ylabel('degree')

        ut.enhance_plot(fig, show, filename, params_dict, directory)

    def set_features(self, proportions, variation, average_events, homophilic=False):
        arr = np.zeros([len(proportions), self.n_nodes])
        self.assortativity = np.zeros(len(proportions))
        for i, feature in enumerate(proportions):
            if homophilic:
                values = self._homophilic_feature(values=[1], max_nodes=int(self.n_nodes * feature))
            else:
                values = self.rng.choice(a=(-1, 1), size=self.n_nodes, p=(1 - feature, feature))
            nx.set_node_attributes(self.network, {i: k for i, k in enumerate(values)}, i)
            self.assortativity[i] = nx.attribute_assortativity_coefficient(self.network, i)
            arr[i] = values
        self.features = arr.T
        logit = 1 / (1 + np.exp(-np.sum(self.features, 1) * variation))
        self.node_average = logit * self.n_nodes * average_events / sum(logit)

    def set_sinusoidal_time(self, end_time, row, omega, phi, num=100):
        t_values = np.linspace(0, end_time, num=num)
        tfs = []
        for n in self.node_average:
            y_values = n * (1 + row * np.sin(omega * t_values + phi)) / end_time
            tfs.append(TimeFunction((t_values, y_values)))
        self.time_functions = tfs

    # TODO Improve
    def plot_node_average(self, show=True, filename=None, params_dict=None, directory='results'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        ax1.hist(self.features, bins=50)
        ax2.hist(self.node_average, bins=50)
        ut.enhance_plot(fig, show, filename, params_dict, directory)
