import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

os.environ['DISPLAY'] = '0'
from src import SimuBaseline

# TODO Turn this into an integration test

n = 1000
features = 5
p = 0.2
variation = 0.5
average_mu = 0.2
r = 10
avg_risk = []
coeffs = []
scores = []

for seed in range(r):
    rng = np.random.default_rng(seed)
    model = LogisticRegression(random_state=seed)

    bs = SimuBaseline(n_nodes=n, network_type='path', seed=seed)
    bs.simulate(feature_proportions=np.ones(features) * p, mu_variation=variation, mu_mean=average_mu)
    avg_risk.append(np.average(bs.node_mu))
    y = rng.binomial(n=1, p=bs.node_mu)
    model.fit(bs.features, y)
    y_pred = model.predict(bs.features)

    coeffs.append(model.coef_[0])
    scores.append(accuracy_score(y, y_pred))

bs.plot_node_mu(show=True)
print(np.average(avg_risk))
# Should be close to average_events
print(np.average(coeffs),np.std(coeffs))
# Should be close to variation
print(np.average(scores))
