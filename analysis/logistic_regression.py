import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import src.inference as infer

n = 1000
r = 5
coeffs = []
scores = []

for seed in range(r):
    rng = np.random.default_rng(seed)
    gender = rng.choice((-1, 1), size=n, p=(0.9,0.1))
    age = rng.choice(range(-2,2), size=n, p=(0.4,0.3,0.2,0.1))
    arrests = rng.choice(range(-2,2), size=n, p=(0.4,0.3,0.2,0.1))
    xb = gender + age + arrests
    risk = 1 / (1 + np.exp(-xb))
    plt.hist(risk); plt.show()
    y = rng.binomial(n=1, p=risk)
    x = np.array([gender, age, arrests]).T

    model = LogisticRegression(random_state=seed)
    model.fit(x, y)
    coeffs.append(np.concatenate([model.coef_[0], model.intercept_]))

    y_pred = model.predict(x)
    # print(confusion_matrix(y, y_pred))

    risk_highest = infer.get_highest_risk_nodes(risk)
    risk_pred = model.predict_proba(x)[:, 1]
    risk_pred_highest = infer.get_highest_risk_nodes(risk_pred)
    # print(confusion_matrix(risk_pred_highest,risk_highest))

    scores.append((accuracy_score(y, y_pred), accuracy_score(risk_pred_highest, risk_highest)))

print(np.average(coeffs, 0), np.std(coeffs, 0))
# [ 0.56168989  0.51481353  0.53003223 -0.01008016] [0.24891125 0.02398502 0.0661345  0.06669541]
print(np.average(scores, 0), np.std(scores, 0))
# [0.7808 0.9873] [0.01017644 0.00863771]
