import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

df_objects = pd.read_csv('data/cb_objects.csv')
df_ipo = pd.read_csv('data/cb_ipos.csv')
df_acquisitions = pd.read_csv('data/cb_acquisitions.csv')

keep =[  'parent_id', 
       'normalized_name', 'category_code',  'founded_at',
        'domain', 'twitter_username',
        'short_description', 'description',
       'overview', 'tag_list', 'country_code', 'state_code', 'city', 'region',
       'first_investment_at', 'last_investment_at', 'investment_rounds',
       'invested_companies', 'first_funding_at', 'last_funding_at',
       'funding_rounds', 'funding_total_usd', 'first_milestone_at',
       'last_milestone_at', 'milestones', 'relationships'
       ]
removed = ['id', 'entity_type', 'entity_id', 'name', 'permalink', 'status', 'closed_at', 'homepage_url',  'logo_url', 'logo_width', 'logo_height','created_by', 'created_at', 'updated_at']

df_joined = df_objects.set_index("id").join(df_ipo.set_index("object_id"), rsuffix="_ipo").join(df_acquisitions.set_index("acquired_object_id"),  rsuffix="_acq")
df_num = df_joined[keep]._get_numeric_data()
df_num = df_num.fillna(df_num.mean())
target = df_joined.apply(lambda x: 0 if (pd.isnull(x['id_acq']) and pd.isnull(x['ipo_id']))  else 1, axis =1)
clf = RandomForestClassifier(max_depth=5, random_state=0, class_weight='balanced')
model = clf.fit(df_num.values, target.values)
y_hat = model.predict(df_num.values)
print(precision_score(target, y_hat))