import pandas as pd
import numpy as np


data = pd.read_csv('results/runs.csv')

inference = data[['Name','events_training','logistic_coeff','hawkes_alpha','hawkes_beta']]
df_inf = inference.groupby(['Name']).agg([np.nanmean,np.nanstd])
summary_inf = df_inf.groupby(level=0,axis=1).apply(lambda x: np.round(x,3).astype(str).apply('$\pm$'.join, 1))
print(summary_inf.to_latex(escape=False))

pred = data[['Name','events_testing','HR1_logistic','HR1_hawkes','HR1_combined']]
df_pred = pred.groupby(['Name']).agg([np.nanmean,np.nanstd])
summary_pred = df_pred.groupby(level=0,axis=1).apply(lambda x: np.round(x,3).astype(str).apply('$\pm$'.join, 1))
print(summary_pred.to_latex(escape=False))