from datetime import datetime, timedelta
from random import choice, choices, randrange

import pandas as pd
import networkx as nx

all_person_ids = [f'OCC_PERSON_OVERALL_ID_{x:03}' for x in range(50)]
all_occurrence_ids = [f'OCCURRENCE_ID_{x:03}' for x in range(25)]

def random_date():
    start_date = datetime.strptime('2016-01-01', '%Y-%m-%d')
    end_date   = datetime.strptime('2020-01-01', '%Y-%m-%d')
    delta = end_date - start_date
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start_date + timedelta(seconds=random_second)

n = 200

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
