import numpy as np
from scipy.optimize import minimize, brute

import src.utilities as ut


class HawkesExpKernelIdentical:
    """Hawkes MLE learner for exponential kernels with many simplifying assumptions.

    Assumptions:
    - The excitation kernel is exponential for the explanations given in section ref{exp_kernel}
    - The baseline $mu$ for all nodes are identical and the hawkes excitation $alpha$ and decay rate $beta$ are the same for all nearest-neighbours. This means that only three parameters need to be determined instead of 3D where D is the number of nodes. This is a necessary assumption in the case where the number of events per node is too small to determine individual hawkes parameters
    - The hawkes excitation $alpha$ and decay rate $beta$ for all other pairs (including self-excitation) is assumed to be zero. This is different to Green et al. who included next and next-next-nearest-neighbours. The results will show that this assumption is valid under the circumstances investigated
    - $alpha$ $>1e-10$ because it cannot be negative
    - $alpha$ $<$ 1, after which point the number of contagion events would explode which we were not considering
    - $beta$ $>$ 2/training time otherwise a large proportion of the contagion events would occur after the training time
    - $beta$ $<$ 5 which is equivalent to a lifetime of 0.2 time units, at which point 99.9% of contagion events would occur within 1 time unit which seemed like a sensible cutoff
    - The baseline is either time-homogeneous or sinusoidal where the sinusoidal parameters are known
    """
    
    def __init__(self, network, verbose=False, seed=None):
        self.network = network
        self.n_nodes = network.number_of_nodes()
        self.verbose = verbose
        self.seed = seed
        self.timestamps = None
        self.alpha_range = None
        self.mu_range = None
        self.beta_range = None
        self.n_training_jumps = None
        self.lambda_mean = None
        self.coef_ = None
        self.mu_est = None
        self.alpha_est = None
        self.beta_est = None
        self.optimise_result = None
        self.training_time = None

    def fit(self, timestamps, training_time, row=0, omega=1, phi=0, Ns=5):
        """Uses Scipy algorithms to minimise the negative of the Hawkes log-likelihood.

        A brute force method first computes the function’s value at each point of a multidimensional grid of points,
        to obtain an initial estimate. The L-BFGS-B algorithm is then used to refine the estimate.
        This is a popular algorithm for minimizing non-linear functions with multiple variables and bound constraints.
        The Hawkes log-likelihood is not jointly concave in alpha, beta, mu.
        We investigated using simulated annealing instead but the minimisation took much longer without getting better
        estimates than L-BFGS-B, for the parameter space investigated.

        :param timestamps:
        :param training_time:
        :param row:
        :param omega:
        :param phi:
        :param Ns:
        :return:
        """
        if len(timestamps) != self.n_nodes:
            raise ValueError('length of timestamps should be the same as number of nodes')
        self.timestamps = timestamps
        self.training_time = training_time
        self.n_training_jumps = np.sum(np.concatenate(timestamps) < self.training_time)
        self.lambda_mean = self.n_training_jumps / (self.n_nodes * self.training_time)
        self._set_ranges()
        print('brute starting')
        print(self.mu_range, self.alpha_range, self.beta_range)
        brute_res = brute(self._ll_multi, full_output=True, finish=0, Ns=Ns,
                          ranges=(self.mu_range, self.alpha_range, self.beta_range),
                          args=(timestamps, self.training_time, row, omega, phi, self.verbose), 
                          workers=-1)
        print('brute done', brute_res[0])
        print('minimize starting')
        ggd_res = minimize(self._ll_multi, x0=brute_res[0], method='Powell',
                           bounds=(self.mu_range, self.alpha_range, self.beta_range),
                           args=(timestamps, self.training_time, row, omega, phi, self.verbose))
        print('minimize done', ggd_res)
        self.brute_result = brute_res
        self.optimise_result = ggd_res
        self._set_coeffs(ggd_res.x)

    def _set_coeffs(self, coeffs):
        self.coef_ = coeffs
        if self.alpha_range[0] < np.round(coeffs[1], 5) < self.alpha_range[1] \
                and self.beta_range[0] < np.round(coeffs[2], 5) < self.beta_range[1]:
            self.mu_est = coeffs[0]
            self.alpha_est = coeffs[1]
            self.beta_est = coeffs[2]
        else:
            self.mu_est = self.lambda_mean
            self.alpha_est = 1e-10
            self.beta_est = 0
        self.label = f'$\\hat\\alpha$:{ut.round_to_n(self.alpha_est, 3)}, $\\hat\\beta_h$:{ut.round_to_n(self.beta_est, 3)}'

    def _set_ranges(self):
        # Assumes mu cannot be much larger than the mean intensity
        self.mu_range = (1e-20, 2 * self.lambda_mean)
        # Assumes alpha cannot be larger than 1 which would result in critical number of events
        self.alpha_range = (1e-20, 1)
        # Assumes the contagion lifetime must be smaller than half of the training time
        # and longer than 1/3 of a time unit
        self.beta_range = (2 / self.training_time, 50)

    def _recursive(self, timestamps, beta, ):
        r_array = np.zeros(len(timestamps))
        for i in range(1, len(timestamps)):
            r_array[i] = np.exp(-beta * (timestamps[i] - timestamps[i - 1])) * (1 + r_array[i - 1])
        return r_array

    def _recursive_multi(self, timestamps, timestamps_neighbors, beta):
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

    def _kernel_int(self, timestamps, runtime, alpha, beta):
        # integrate kernel until runtime and sum
        kernel_int = - alpha * np.sum(np.exp(-beta * (runtime - timestamps)) - 1)
        return kernel_int

    def _sinusoidal_comp(self, timestamps, mu, runtime, row, omega, phi):
        # calculate sinusoidal component of baseline at each timestamp
        sinusoidal_func = mu * row * np.sin(omega * timestamps + phi)
        # integrate sinusoidal component of baseline until runtime
        sinusoidal_int = - mu * row * (1 / omega) * (np.cos(omega * runtime + phi) - np.cos(phi))
        return sinusoidal_func, sinusoidal_int

    def _log_likelihood(self, timestamps, mu, alpha, beta, runtime=None, row=0, omega=1, phi=0,
                        timestamps_neighbors=None):
        if runtime is None:
            runtime = timestamps[-1]

        if timestamps_neighbors is None:
            kernel_int = self._kernel_int(timestamps, runtime, alpha, beta)
            r_array = self._recursive(timestamps, beta)
        else:
            kernel_int = self._kernel_int(np.concatenate(timestamps_neighbors), runtime, alpha, beta)
            r_array = np.sum(self._recursive_multi(timestamps, timestamps_neighbors, beta, ), -1)

        sinusoidal_func, sinusoidal_int = self._sinusoidal_comp(timestamps, mu, runtime, row, omega, phi)
        # log-likelihood that each individual was not infected at all other times
        ll_events_occured = np.sum(np.log(mu + sinusoidal_func + alpha * beta * r_array))
        # log-likelihood of every infection event that did occur
        ll_events_not_occured = mu * runtime + sinusoidal_int + kernel_int
        return ll_events_occured - ll_events_not_occured

    def _log_likelihood_multi(self, timestamps, mu, alpha, beta, runtime=None, row=0, omega=1, phi=0):
        ll_multi = 0
        for node in self.network.nodes:
            node_ts = timestamps[node][timestamps[node] <= runtime]
            node_ts_neighbors = [timestamps[i][timestamps[i] <= runtime] for i in self.network.neighbors(node)]
            ll_multi += self._log_likelihood(node_ts, mu, alpha, beta, runtime, row, omega, phi, node_ts_neighbors)
        if self.verbose:
            print(f"mu: {mu}, alpha: {alpha}, beta: {beta}, ll: {ll_multi}")
        return ll_multi

    def _ll_multi(self, params, *args, ):
        mu, alpha, beta = params
        timestamps, runtime, row, omega, phi, verbose = args
        return -self._log_likelihood_multi(timestamps, mu, alpha, beta, runtime, row, omega, phi, )

    def _ll_fixed_beta_multi(self, params, *args):
        mu, alpha, = params
        timestamps, runtime, beta, row, omega, phi, verbose = args
        return -self._log_likelihood_multi(timestamps, mu, alpha, beta, runtime, row, omega, phi, )

    def _ll_fixed_mu_alpha_multi(self, params, *args):
        beta, = params
        timestamps, runtime, mu, alpha, row, omega, phi, verbose = args
        return -self._log_likelihood_multi(timestamps, mu, alpha, beta, runtime, row, omega, phi, )

    def predict_proba(self, times, alpha=None, beta=None):
        """Calculates the exponential kernel intensity component of the Hawkes process at specified times.

        The baseline contribution is ignored since the learner assumes all nodes have the same baseline.
        The kernel intensity is calculated for every time specified to return a Dxtimes matrix.
        The kernel intensity can then be interpreted as the time-dependent Hawkes model risk.

        :param times:
        :param alpha:
        :param beta:
        :return:
        """
        node_risk = np.zeros([len(times), self.n_nodes])
        if alpha is None:
            alpha = self.alpha_est
        if beta is None:
            beta = self.beta_est
        for node in self.network.nodes:
            print('node: ', node)
            node_ts_neighbors = np.concatenate([self.timestamps[i] for i in self.network.neighbors(node)])
            # TODO Limit to multiple of lifetime?
            for i, time in enumerate(times):
                values = node_ts_neighbors[node_ts_neighbors < time]
                node_risk[i, node] = alpha * beta * np.sum(np.exp(-beta * (time - values)))
        return node_risk
