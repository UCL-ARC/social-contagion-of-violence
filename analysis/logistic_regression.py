import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

n = 150
r = 10
coeffs = []
scores = []

for seed in range(r):
    rng = np.random.default_rng(seed)
    gender = (rng.choice((0, 1), n) - 0.5)
    xb = 0.5 * gender + error
    risk = 1 / (1 + np.exp(-xb))
    plt.hist(risk); plt.show()
    y = rng.binomial(n=1, p=risk)
    x = np.array([gender, age]).T

    model = LogisticRegression(random_state=seed)
    model.fit(x, y)
    coeffs.append(np.concatenate([model.coef_[0], model.intercept_]))

    y_pred = model.predict(x)
    print(confusion_matrix(y, y_pred))
    scores.append(accuracy_score(y,y_pred))

print(np.average(coeffs,0), np.std(coeffs,0))
print(np.average(scores,0))