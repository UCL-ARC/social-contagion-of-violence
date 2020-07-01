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


# Copied from https://lmfit.github.io/lmfit-py/examples/example_brute.html
# TODO Fix following function
def plot_results_brute(result, best_vals=True, varlabels=None,
                       output=None):
    """Visualize the result of the brute force grid search.
    The output file will display the chi-square value per parameter and contour
    plots for all combination of two parameters.

    Inspired by the `corner` package (https://github.com/dfm/corner.py).

    Parameters
    ----------
    result : :class:`~lmfit.minimizer.MinimizerResult`
        Contains the results from the :meth:`brute` method.

    best_vals : bool, optional
        Whether to show the best values from the grid search (default is True).

    varlabels : list, optional
        If None (default), use `result.var_names` as axis labels, otherwise
        use the names specified in `varlabels`.

    output : str, optional
        Name of the output PDF file (default is 'None')
    """
    npars = len(result.var_names)
    _fig, axes = plt.subplots(npars, npars)

    if not varlabels:
        varlabels = result.var_names
    if best_vals and isinstance(best_vals, bool):
        best_vals = result.params

    for i, par1 in enumerate(result.var_names):
        for j, par2 in enumerate(result.var_names):

            # parameter vs chi2 in case of only one parameter
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=3)
                axes.set_ylabel(r'$\chi^{2}$')
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on top
            elif i == j and j < npars - 1:
                if i == 0:
                    axes[0, 0].axis('off')
                ax = axes[i, j + 1]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.unique(result.brute_grid[i]),
                        np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        'o', ms=3)
                ax.set_ylabel(r'$\chi^{2}$')
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('right')
                ax.set_xticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on the left
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        np.unique(result.brute_grid[i]), 'o', ms=3)
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i])
                if i != npars - 1:
                    ax.set_xticks([])
                elif i == npars - 1:
                    ax.set_xlabel(r'$\chi^{2}$')
                if best_vals:
                    ax.axhline(best_vals[par1].value, ls='dashed', color='r')

            # contour plots for all combinations of two parameters
            elif j > i:
                ax = axes[j, i + 1]
                red_axis = tuple([a for a in range(npars) if a not in (i, j)])
                X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                   np.unique(result.brute_grid[j]))
                lvls1 = np.linspace(result.brute_Jout.min(),
                                    np.median(result.brute_Jout) / 2.0, 7, dtype='int')
                lvls2 = np.linspace(np.median(result.brute_Jout) / 2.0,
                                    np.median(result.brute_Jout), 3, dtype='int')
                lvls = np.unique(np.concatenate((lvls1, lvls2)))
                ax.contourf(X.T, Y.T, np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            lvls, norm=LogNorm())
                ax.set_yticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')
                    ax.axhline(best_vals[par2].value, ls='dashed', color='r')
                    ax.plot(best_vals[par1].value, best_vals[par2].value, 'rs', ms=3)
                if j != npars - 1:
                    ax.set_xticks([])
                elif j == npars - 1:
                    ax.set_xlabel(varlabels[i])
                if j - i >= 2:
                    axes[i, j].axis('off')

    if output is not None:
        plt.savefig(output)
