from datetime import datetime, timedelta
from random import choice, choices, randrange

import pandas as pd
import numpy as np
import networkx as nx

import src

all_person_ids = [f'OCC_PERSON_OVERALL_ID_{x:03}' for x in range(50)]
all_occurrence_ids = [f'OCCURRENCE_ID_{x:03}' for x in range(25)]

def random_date():
    start_date = datetime.strptime('2016-01-01', '%Y-%m-%d')
    end_date   = datetime.strptime('2020-01-01', '%Y-%m-%d')
    delta = end_date - start_date
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start_date + timedelta(seconds=random_second)

n = 100

person_ids       = [choice(all_person_ids) for _ in range(n)]
occurrence_ids   = [choice(all_occurrence_ids) for _ in range(n)]
datetimes        = [random_date() for _ in range(n)]
is_violent_crime = [choices([True, False], weights=[0.2, 0.8])[0] for _ in range(n)]

dummy_df = pd.DataFrame({'ID_LINKED_ON_NICHE_PNC_NAME_DOB_POSTCODE': person_ids, 
                         'OCCURRENCE_ID': occurrence_ids, 
                         'REPORTED_DATE': datetimes, 
                         'VIOLENT_CRIME_FLAG': is_violent_crime})

B = nx.from_pandas_edgelist(dummy_df, 'ID_LINKED_ON_NICHE_PNC_NAME_DOB_POSTCODE', 'OCCURRENCE_ID')
G = nx.bipartite.projected_graph(B, dummy_df['ID_LINKED_ON_NICHE_PNC_NAME_DOB_POSTCODE'])

print(G.order())
print(len(dummy_df['ID_LINKED_ON_NICHE_PNC_NAME_DOB_POSTCODE'].unique()))

dummy_df['DAY_INT'] = (dummy_df['REPORTED_DATE'].max() - dummy_df['REPORTED_DATE']).dt.days

train_test_threshold = int(np.quantile(dummy_df['DAY_INT'], 0.7))
last_day_in_data = dummy_df['DAY_INT'].max()

t = []
for n in G: 
    mask = (dummy_df['ID_LINKED_ON_NICHE_PNC_NAME_DOB_POSTCODE']  == n) & (dummy_df['VIOLENT_CRIME_FLAG'])
    t.append(np.array(dummy_df.loc[mask, 'DAY_INT']))

model_contagion = src.HawkesExpKernelIdentical(G, verbose=True)
model_contagion.fit(t, training_time=train_test_threshold, row=0, omega=1, phi=0)
preds = model_contagion.predict_proba(range(train_test_threshold, last_day_in_data))
print(preds)
