import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

n = 1000
r = 10
coeffs = []
scores = []
scaling = 1

for seed in range(r):
    rng = np.random.default_rng(seed)
    gender = rng.choice((-1, 1), size=n, p=(0.9,0.1))
    age = rng.choice((-1, 1), size=n, p=(0.9,0.1))
    arrests = rng.choice((-1, 1), size=n, p=(0.9,0.1))
    xb = gender + age + arrests
    risk = 1 / (1 + np.exp(-xb*scaling))
    plt.hist(risk); plt.show()
    y = rng.binomial(n=1, p=risk)
    x = np.array([gender, age, arrests]).T

    model = LogisticRegression(random_state=seed)
    model.fit(x, y)
    coeffs.append(np.concatenate([model.coef_[0], model.intercept_]))
    y_pred = model.predict(x)
    scores.append(accuracy_score(y, y_pred))

print(np.average(coeffs, 0), np.std(coeffs, 0))
# [0.93457335 1.01115506 1.01225805 0.01431867] [0.11067048 0.10161891 0.10335749 0.10000723]
print(np.average(scores, 0), np.std(scores, 0))
# 0.8924 0.008475848040166843

