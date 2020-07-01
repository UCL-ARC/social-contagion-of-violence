import numpy as np
from scipy.optimize import minimize, basinhopping, brute
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels, HawkesExpKern
from tick.base import TimeFunction
import src.inference as infer
import src.homophily as ho

########################################################################################################################
# INITIALISE

mu = 0.5
a = 0.2
b = 0.3
rt = 1000
row = 0.75
omega = 0.05
phi = np.pi

#######################################################################################################################
# FIXED BASELINE

simu = SimuHawkesExpKernels([[a]], b, [mu], rt, seed=0)
simu.simulate()
t = simu.timestamps[0]

# Brute search
rranges = (slice(10 ** -6, 1, 0.2), slice(0, 1, 0.2), slice(0, 0.5, 0.1),)
brute_res = brute(infer.crit, ranges=rranges, args=(t, rt, 0, 1, 0), full_output=True, finish=0, )
print(brute_res[0])
# array([0.400001, 0.4     , 0.1     ])

# Tick Learner - fixed beta
learner = HawkesExpKern(brute_res[0][2])
learner.fit(simu.timestamps)
print(learner.coeffs)
# [0.41025476 0.30842368]

# Gradient descent - fixed beta
gd_res = minimize(infer.crit_fixed_beta, x0=brute_res[0][0:2], args=(t, rt, brute_res[0][2], 0, 1, 0),
                  method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None),))
print(gd_res.x)
# [0.41954164 0.29106901]

# Simulated annealing - fixed mu and alpha
bh_res = basinhopping(infer.crit_fixed_mu_alpha, x0=brute_res[0][2], disp=True,
                      minimizer_kwargs={'args': (t, rt) + tuple(gd_res.x) + (0, 1, 0), 'bounds': ((1e-10, 1),)})
print(bh_res.x)
# [0.13512682]

# Global gradient descent
ggd_res = minimize(infer.crit, x0=brute_res[0], args=(t, rt, 0, 1, 0), method='L-BFGS-B',
                   bounds=((1e-10, None), (1e-10, None), (1e-10, None)))
print(ggd_res.x)
# x: [0.43835601, 0.25823677, 0.14769127]

print(infer.log_likelihood(t, mu, a, b, rt))
print(infer.log_likelihood(t, *brute_res[0], rt))
print(infer.log_likelihood(t, *gd_res.x, bh_res.x[0], rt))
print(infer.log_likelihood(t, *ggd_res.x, rt))
# -898.9579084145716
# -899.8068750243427
# -897.8072093278508
# -897.7686269881596


#######################################################################################################################
# SINUSOIDAL BASELINE

t_values = np.linspace(0, rt, num=100)
y_values = mu * (1 + row * np.sin(omega * t_values + phi))
sinu_baseline = TimeFunction((t_values, y_values))
ho.plot_time_functions(sinu_baseline)

simu_sinu = SimuHawkesExpKernels([[a]], b, [sinu_baseline], rt, seed=0)
simu_sinu.simulate()
sinu_t = simu_sinu.timestamps[0]

# Brute search
sinu_brute_res = brute(infer.crit, ranges=rranges, args=(sinu_t, rt, row, omega, phi), full_output=True,
                       finish=0, )
print(sinu_brute_res[0])
# [0.400001 0.4      0.2     ]

# Global gradient descent
sinu_ggd_res = minimize(infer.crit, x0=sinu_brute_res[0], args=(sinu_t, rt, row, omega, phi),
                        bounds=((1e-10, None), (1e-10, None), (1e-10, None)))
print(sinu_ggd_res.x)
# [0.41825757 0.32442451 0.17155854]

# Global gradient descent but using a constant a baseline
# The estimated parameters are further from the actual parameters which is a good indication the function works
sinu_ggd_res = minimize(infer.crit, x0=sinu_brute_res[0], args=(sinu_t, rt, 0, 1, 0 ),
                        bounds=((1e-10, None), (1e-10, None), (1e-10, None)))
print(sinu_ggd_res.x)
# [0.19459373 0.68914562 0.15929386]
