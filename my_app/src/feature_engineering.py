from collections import defaultdict
import json
import pandas as pd
def new_x(x):
    nut = json.loads(x['nutrition'].replace('\'', '\"').replace('\"s ', '\'s '))['nutrients']
    for obj in nut:
        title = obj['title']
        amount = obj['amount']
        percentOfDailyNeeds = obj['percentOfDailyNeeds']
        unit = obj['unit']
        x[title+'_amount'] = (amount)
        x[title+'_percentOfDailyNeeds'] = (percentOfDailyNeeds)
        x[title+'_unit'] = (unit)
    return x

def get_overall_nutrition_about_meal(df):
    old_columns = list(df.columns)
    old_columns.remove('title')
    new_df = df.apply(new_x, axis =1)
    return new_df.drop(columns=old_columns, axis=1)
