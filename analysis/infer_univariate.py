import numpy as np
from scipy.optimize import minimize, basinhopping, brute
import os

os.environ['DISPLAY'] = '0'  # Workaround to ensure tick does not overwrite the matplotlib back_end
from tick.hawkes import SimuHawkesExpKernels, HawkesExpKern
from tick.base import TimeFunction
import src.inference as infer
import src.homophily as ho
import src.analyse as an

########################################################################################################################
# INITIALISE

mu = 0.5
a = 0.2
b = 0.2
rt = 2000
row = 0.75
omega = 0.05
phi = np.pi
n_realizations = 10

#######################################################################################################################
# FIXED BASELINE

simu = SimuHawkesExpKernels([[a]], b, [mu], rt, seed=0)
simu.simulate()
timestamps = an.repeat_simulations(simu, n_realizations)
print([len(t[0]) for t in timestamps])

coeffs_tick = np.zeros([n_realizations, 2])
coeffs_ll = np.zeros([n_realizations, 2])
coeffs_global = np.zeros([n_realizations, 3])
coeffs_anneal = np.zeros([n_realizations, 3])
actual = np.array([mu, a, b])
rranges = (slice(10 ** -6, 1, 0.1), slice(0, 1, 0.1), slice(0, 1, 0.1),)

for i, t in enumerate(timestamps):
    # Brute search
    brute_res = brute(infer.crit, ranges=rranges, args=(t[0], rt, 0, 1, 0), full_output=True, finish=0, )
    # print(brute_res[0])
    # array([0.400001, 0.4     , 0.1     ])

    # Tick Learner - known beta
    learner = HawkesExpKern(b)
    learner.fit(t)
    coeffs_tick[i] = learner.coeffs
    print(np.average(coeffs_tick,0), np.std(coeffs_tick,0))
    # [0.48975647 0.21470438] [0.03757358 0.05970369]

    # Gradient descent - known beta
    gd_res = minimize(infer.crit_fixed_beta, x0=brute_res[0][0:2], args=(t[0], rt, b, 0, 1, 0),
                      method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None),))
    coeffs_ll[i] = gd_res.x
    print(np.average(coeffs_ll,0), np.std(coeffs_ll,0))
    # [0.49019813 0.21367062][0.03721219 0.06053413]

    # Global gradient descent
    ggd_res = minimize(infer.crit, x0=brute_res[0], args=(t[0], rt, 0, 1, 0),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, 1), (1e-10, None)))
    coeffs_global[i] = ggd_res.x
    print(np.average(coeffs_global,0), np.std(coeffs_global,0))
    # [0.48314747 0.22732935 0.54965216][0.07954101    0.12569939    0.79258957]

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
sinu_ggd_res = minimize(infer.crit, x0=sinu_brute_res[0], args=(sinu_t, rt, 0, 1, 0),
                        bounds=((1e-10, None), (1e-10, None), (1e-10, None)))
print(sinu_ggd_res.x)
# [0.19459373 0.68914562 0.15929386]
