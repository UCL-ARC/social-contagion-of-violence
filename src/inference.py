import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def _recursive(timestamps, beta, ):
    r_array = np.zeros(len(timestamps))
    for i in range(1, len(timestamps)):
        r_array[i] = np.exp(-beta * (timestamps[i] - timestamps[i - 1])) * (1 + r_array[i - 1])
    return r_array


def _recursive_multi(timestamps, timestamps_neighbors, beta):
    r_array = np.zeros((len(timestamps), len(timestamps_neighbors)))
    if len(timestamps) > 0:
        for n in range(len(timestamps_neighbors)):
            tsn = np.where(timestamps_neighbors[n] < timestamps[0])
            r_array[0, n] = np.sum(np.exp(-beta * (timestamps[0] - timestamps_neighbors[n][tsn])))
        for k in range(1, len(timestamps)):
            for n in range(len(timestamps_neighbors)):
                tsn = np.where(np.logical_and(timestamps_neighbors[n] >= timestamps[k - 1],
                                              timestamps_neighbors[n] < timestamps[k]))
                r_array[k, n] = np.exp(-beta * (timestamps[k] - timestamps[k - 1])) * r_array[k - 1, n] + \
                                np.sum(np.exp(-beta * (timestamps[k] - timestamps_neighbors[n][tsn])))
    return r_array


def _kernel_int(timestamps, runtime, alpha, beta):
    # integrate kernel until runtime and sum
    kernel_int = - alpha * np.sum(np.exp(-beta * (runtime - timestamps)) - 1)
    return kernel_int


def _sinusoidal_comp(timestamps, mu, runtime, row, omega, phi):
    # calculate sinusoidal component of baseline at each timestamp
    sinusoidal_func = mu * row * np.sin(omega * timestamps + phi)
    # integrate sinusoidal component of baseline until runtime
    sinusoidal_int = - mu * row * (1 / omega) * (np.cos(omega * runtime + phi) - np.cos(phi))
    return sinusoidal_func, sinusoidal_int


def log_likelihood(timestamps, mu, alpha, beta, runtime=None, row=0, omega=1, phi=0, timestamps_neighbors=None):
    if runtime is None:
        runtime = timestamps[-1]

    if timestamps_neighbors is None:
        kernel_int = _kernel_int(timestamps, runtime, alpha, beta)
        r_array = _recursive(timestamps, beta)
    else:
        kernel_int = _kernel_int(np.concatenate(timestamps_neighbors), runtime, alpha, beta)
        r_array = np.sum(_recursive_multi(timestamps, timestamps_neighbors, beta, ), -1)

    sinusoidal_func, sinusoidal_int = _sinusoidal_comp(timestamps, mu, runtime, row, omega, phi)
    # log-likelihood that each individual was not infected at all other times
    ll_events_occured = np.sum(np.log(mu + sinusoidal_func + alpha * beta * r_array))
    # log-likelihood of every infection event that did occur
    ll_events_not_occured = mu * runtime + sinusoidal_int + kernel_int
    return ll_events_occured - ll_events_not_occured


def log_likelihood_multi(g, timestamps, mu, alpha, beta, runtime=None, row=0, omega=1, phi=0, verbose=False):
    ll_multi = 0
    for node in g.nodes:
        node_ts = timestamps[node]
        node_ts_neighbors = [timestamps[i] for i in g.neighbors(node)]
        ll_multi += log_likelihood(node_ts, mu, alpha, beta, runtime, row, omega, phi, node_ts_neighbors)
    if verbose:
        print(f"mu: {mu}, alpha: {alpha}, beta: {beta}, ll: {ll_multi}")
    return ll_multi


def crit(params, *args, ):
    mu, alpha, beta = params
    timestamps, runtime, row, omega, phi = args
    return -log_likelihood(timestamps, mu, alpha, beta, runtime, row, omega, phi)


def crit_multi(params, *args, ):
    mu, alpha, beta = params
    g, timestamps, runtime, row, omega, phi, verbose = args
    return -log_likelihood_multi(g, timestamps, mu, alpha, beta, runtime, row, omega, phi, verbose)


def crit_fixed_beta(params, *args):
    mu, alpha, = params
    timestamps, runtime, beta, row, omega, phi, = args
    return -log_likelihood(timestamps, mu, alpha, beta, runtime, row, omega, phi)


def crit_fixed_beta_multi(params, *args):
    mu, alpha, = params
    g, timestamps, runtime, beta, row, omega, phi, verbose = args
    return -log_likelihood_multi(g, timestamps, mu, alpha, beta, runtime, row, omega, phi, verbose)


def crit_fixed_mu_alpha(params, *args):
    beta, = params
    timestamps, runtime, mu, alpha, row, omega, phi = args
    return -log_likelihood(timestamps, mu, alpha, beta, runtime, row, omega, phi, )


def crit_fixed_mu_alpha_multi(params, *args):
    beta, = params
    g, timestamps, runtime, mu, alpha, row, omega, phi, verbose = args
    return -log_likelihood_multi(g, timestamps, mu, alpha, beta, runtime, row, omega, phi, verbose)


def contagion_risk(g, timestamps, alpha, beta, times):
    node_risk = np.zeros([len(times), g.number_of_nodes()])
    for node in g.nodes:
        node_ts_neighbors = np.concatenate([timestamps[i] for i in g.neighbors(node)])
        for i, time in enumerate(times):
            values = node_ts_neighbors[np.where(node_ts_neighbors < time)]
            node_risk[i, node] = alpha * beta * np.sum(np.exp(-beta * (time - values)))
    return node_risk


def get_highest_risk_nodes(node_risk, percent=1):
    n = int(np.ceil(len(node_risk) * percent / 100))
    value = -np.sort(-node_risk)[n - 1]
    highest_risk = np.zeros([len(node_risk)])
    for node, risk in enumerate(node_risk):
        if risk >= value:
            highest_risk[node] = 1
    return highest_risk
