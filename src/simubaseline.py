import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tick.base import TimeFunction

import src.utilities as ut


class SimuBaseline:
    """ Takes a network as an input, or parameters for generating a network,
    and simulates the node features and baseline according to the configuration parameters.
    It can also plot the distribution of node features and visualise the network.
    """

    def __init__(self, n_nodes=None, network_type=None, seed=None, network=None):
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
        self._set_network(network)
        self._set_adjacency()

    def _set_network(self, network):
        if self.network_type is not None and self.n_nodes is not None:
            if self.network_type == 'BA':
                # Create a graph with power-law distribution
                g = nx.barabasi_albert_graph(self.n_nodes, 1, seed=self.seed)
            elif self.network_type == 'WS':
                # Create a graph with clustering and small-world
                g = nx.watts_strogatz_graph(self.n_nodes, 4, 0.5, seed=self.seed)
            elif self.network_type == 'path':
                g = nx.path_graph(self.n_nodes)
            elif self.network_type == 'star':
                g = nx.star_graph(self.n_nodes)
            else:
                raise ValueError('Unknown network type')
            self.network = g
        elif network is not None:
            if isinstance(network, nx.Graph):
                self.network = network

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
    def plot_network(self, n=200, show=True, filename=None, params_dict=None, directory='results'):
        fig = plt.figure(figsize=(8, 5))
        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0:-1])
        ax2 = fig.add_subplot(gs[0, -1])

        # plot network
        g = nx.subgraph(self.network, range(n))
        pos = nx.spiral_layout(g)
        nx.draw_networkx_edges(g, pos, alpha=0.3, ax=ax1)
        node_color = [k for i, k in g.nodes.data(-1)]
        nc = nx.draw_networkx_nodes(g, pos=pos, node_size=4, node_color=node_color, cmap=plt.cm.jet, ax=ax1,
                                    label=f"Node baseline (assortativity = {np.round(self.assortativity[-1], 3)})")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(nc, ax=ax1, orientation='horizontal', cax=cax)
        cbar.ax.locator_params(nbins=5, tight=True)
        ax1.legend(loc='lower left')
        ax1.set_title(f'Network Diagram of nodes 0-{n} and edges between')
        ax1.axis('off')

        # plot degree rank
        degree_sequence = sorted([d for n, d in self.network.degree()], reverse=True)
        ax2.plot(degree_sequence, marker='.')
        ax2.set_title('Degree rank plot')
        ax2.set_xlabel(f'Rank')
        ax2.set_ylabel('Degree')
        plt.tight_layout()
        ut.enhance_plot(fig, show, filename, params_dict, directory)

    def simulate(self, mu_mean, mu_variation=0, features=None, feature_proportions=None, homophilic=False, end_time=1,
                 row=0, omega=1, phi=0, ):
        self._set_features(features, feature_proportions, homophilic)
        self._set_node_mu(mu_mean, mu_variation)
        self._set_assortativity()
        self._set_sinusoidal_time(end_time, row, omega, phi, )

    def _set_features(self, features=None, feature_proportions=None, homophilic=False, ):
        if features is None and feature_proportions is None:
            raise ValueError('features or feature_prob must be provided')
        if features is not None:
            self.features = features
        else:
            if homophilic:
                self.features = np.zeros(([self.n_nodes, len(feature_proportions)]))
                for i, p in enumerate(feature_proportions):
                    self.features[:, i] = self._homophilic_feature(values=[1], max_nodes=int(self.n_nodes * p))
            else:
                self.features = self.rng.binomial(n=1, p=feature_proportions,
                                                  size=(self.n_nodes, len(feature_proportions)))

    # TODO improve how node mu is calculated and corrects for negative values
    def _set_node_mu(self, mu_mean, mu_variation):
        self.sum_features = np.sum(self.features - np.average(self.features), 1)
        self.node_mu = self.sum_features * mu_variation * mu_mean + mu_mean

    def _set_assortativity(self):
        """ Calculates assortativity of features and node baseline.

        Assumes features to be 0,1 attributes. Calculates assortativity of node baseline by normalising, scaling by 1000
        and convert to int because numeric_assortativity expects integer.
        """
        self.assortativity = np.zeros(len(self.features[0]) + 1)
        for i in range(len(self.features[0])):
            nx.set_node_attributes(self.network, dict(enumerate(self.features[:, i])), i)
            self.assortativity[i] = nx.attribute_assortativity_coefficient(self.network, i)
        node_mu_norm = (self.node_mu * 1000 / np.max(self.node_mu)).astype(int)
        nx.set_node_attributes(self.network, dict(enumerate(node_mu_norm)), -1)
        self.assortativity[-1] = nx.numeric_assortativity_coefficient(self.network, -1)

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

        ax1.scatter(self.sum_features, self.node_mu)
        ax1.set_title('a) Sum of node features against baseline')
        ax1.set_ylabel('Node baseline $\\mu^k$')
        ax1.set_xlabel('Sum of normalised node features')

        ax2.hist(self.node_mu)
        ax2.axvline(x=np.nanmean(self.node_mu), linewidth=2, color='r',
                    label=f'Mean baseline: {ut.round_to_n(np.nanmean(self.node_mu), 3)}')
        ax2.legend()
        ax2.set_title('b) Histogram of node baseline')
        ax2.set_xlabel('Node baseline $\\mu^k$')
        ax2.set_ylabel('Frequency')

        plt.tight_layout()
        ut.enhance_plot(fig, show, filename, params_dict, directory)
