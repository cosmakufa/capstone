import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split, cross_val_score
import matplotlib.pyplot as plt



def get_data_merged():
    df_round = get_funding_rounds_df()
    df_object = get_objects_df()
    round_object = pd.merge(left=df_round, right=df_object, left_on='object_id', right_on='id', how='left')
    return round_object

def my_roc_curve(y_true, y_pred):
    xaxis= np.linspace(0, 1, 100)
    #result = np.zeros(xaxis.shape)
    result = {"f1": np.zeros(xaxis.shape), "precision": np.zeros(xaxis.shape), "recall": np.zeros(xaxis.shape)}
    for i,x in enumerate(xaxis):
        y_pred2 = y_pred.copy()
        y_pred2[(y_pred2 >= x)] = 1
        y_pred2[(y_pred2 < x)] = 0
        result["f1"][i] = f1_score(y_true,y_pred2)
        result["precision"][i] = precision_score(y_true,y_pred2)
        result["recall"][i] = recall_score(y_true,y_pred2)
    return (xaxis, result)

def best_train_model():
    df = get_investor_and_objects()
    df= get_all_targets(df)
    df2 = df.drop(columns=['id', 'funding_round_code', 'funded_at'], axis=1)
    target = df2.pop('target')
    X_train, X_test, y_train, y_test = train_test_split(df2, target, test_size=0.25, random_state=1, stratify=target)
    X_train.fillna(X_train.mean(), inplace=True)
    poly = PolynomialFeatures(3, interaction_only=True)
    poly_train = poly.fit_transform(X_train)
    X_test.fillna(X_train.mean(), inplace=True)
    poly_test = poly.transform(X_test)
#     parameters = {'class_weight'= 'balanced',
#  'max_depth'= 11,
#  'n_estimators'= 81,
#  'random_state'= 0}
    model = RandomForestClassifier(class_weight='balanced', max_depth= 11, n_estimators= 81,
 random_state= 0)
    #clf = GridSearchCV(model, parameters, verbose=1)
    model.fit(X_train, y_train)
    y_hat =np.array(list(list(zip(*model.predict_proba(X_test)))[1]))
    x, result = my_roc_curve(y_test, y_hat)
    plt.plot(x, result['f1'], label='f')
    plt.plot(x, result['precision'], label='p')
    plt.plot(x, result['recall'], label='r')
    plt.legend()

def train_model():
    df = get_investor_and_objects()
    df= get_all_targets(df)
    df2 = df.drop(columns=['id', 'funding_round_code', 'funded_at'], axis=1)
    target = df2.pop('target')
    X_train, X_test, y_train, y_test = train_test_split(df2, target, test_size=0.25, random_state=1, stratify=target)
    X_train.fillna(X_train.mean(), inplace=True)
    poly = PolynomialFeatures(3, interaction_only=True)
    poly_train = poly.fit_transform(X_train)
    X_test.fillna(X_train.mean(), inplace=True)
    poly_test = poly.transform(X_test)
    parameters = {"n_estimators": range(1, 500,20), "max_depth":range(1, 20,2), "random_state":[0], "class_weight":['balanced']}
    model = RandomForestClassifier()
    clf = GridSearchCV(model, parameters, verbose=2,n_jobs=5, scoring='f1')
    clf.fit(X_train, y_train)
    y_hat =np.array(list(list(zip(*clf.predict_proba(X_test)))[1]))
    x, result = my_roc_curve(y_test, y_hat)
    plt.plot(x, result['f1'], label='f')
    plt.plot(x, result['precision'], label='p')
    plt.plot(x, result['recall'], label='r')
    plt.legend()
    return X_train, X_test, y_train, y_test, clf

def get_X_target():
    df = get_investor_and_objects()
    df= get_all_targets(df)
    df2 = df.drop(columns=['id', 'funding_round_code', 'funded_at'], axis=1)
    target = df2.pop('target')
    df2.fillna(df2.mean(), inplace=True)
    return df2, target


def get_investor_and_objects():
    df = get_unique_investors()
    df_object = get_objects_df()
    round_object = pd.merge(left=df, right=df_object, left_on='id', right_on='id', how='left')
    return round_object

def unique_investors(x, investors):
    round_id = x['funding_round_id']
    investor = x['investor_object_id']
    investors[round_id].add(investor)
    return len(investors[round_id])

def get_unique_investors():
    df_round = get_total_raised_so_far()
    df_invest = get_investments_df()
    round_invest = pd.merge(left=df_round[['object_id','funded_at']], right=df_invest, left_on='object_id', right_on='funded_object_id', how='right')
    round_invest.sort_values('funded_at', inplace=True) 
    investors_counter = defaultdict(set)
    round_invest["unique_investor"] = round_invest.apply(unique_investors,investors=investors_counter, axis=1)
    grouping = round_invest.groupby(['funding_round_id'])['object_id','unique_investor'].max()
    grouping.reset_index(inplace=True)
    result = pd.merge(left=df_round, right=grouping, left_on='funding_round_id', right_on='funding_round_id', how='left')
    result.drop(columns=['object_id_y'], inplace=True)
    result.columns =  ['funding_round_id','id','funded_at','funding_round_code','raised_amount_usd',
    'participants','is_first_round','round_companies','round_funds','round_angels','total_companies',
    'total_funds', 'total_angels','total_raised', 'unique_investor']
    df = result
    keep = [  
    'grant',
    'convertible',
    'angel',
    'seed',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'partial',
    'private_equity',
    'debt_round'] 
    code_mapping = {x:i for i,x in enumerate(keep)}
    df = df[df['funding_round_code'].isin(keep)]
    df['mapped_code'] = df['funding_round_code'].apply(lambda x: code_mapping[x])
    return df

def total_raised(x, investor):
    object_id = x['object_id']
    investor[object_id] += x["raised_amount_usd"]
    return investor[object_id]

def get_total_raised_so_far():
    df_round = get_investor_total_type_so_far()
    raised = Counter()
    df_round['total_raised'] = df_round.apply(total_raised,investor=raised,axis=1)
    return df_round

def total_investor(x, investor, r_type):
    object_id = x['object_id']
    investor[object_id] += x[r_type]
    return investor[object_id]

def get_investor_total_type_so_far():
    df_round = get_investor_type_in_round()
    companys = Counter()
    funds = Counter()
    angels = Counter()
    df_round.sort_values("funded_at", inplace=True, ascending=True)
    df_round['total_companies'] = df_round.apply(total_investor,investor=companys,r_type="round_companies", axis=1)
    df_round['total_funds'] = df_round.apply(total_investor,investor=funds,r_type="round_funds", axis=1 )
    df_round['total_angels'] = df_round.apply(total_investor,investor=angels,r_type="round_angels", axis=1 )
    return df_round

def add_investor(x, investor):
    round_id = x['funding_round_id']
    return investor[round_id]

def get_investor_type_in_round():
    df_round = get_funding_rounds_df()
    df_invest = get_investments_df()
    companys = Counter()
    funds = Counter()
    angels = Counter()
    others = Counter()
    nans = Counter()
    for i, row in df_invest.iterrows():
        investor = row['investor_type']
        round_id = row['funding_round_id']
        if investor == 'company':
            companys[round_id] += 1
        elif investor == 'angel':
            angels[round_id] += 1
        elif investor == 'fund':
            funds[round_id] += 1
        elif pd.isna(investor):
            nans[round_id] += 1
        else:
            others[round_id] += 1
    df_round['round_companies'] = df_round.apply(add_investor,investor=companys, axis=1)
    df_round['round_funds'] = df_round.apply(add_investor,investor=funds, axis=1 )
    df_round['round_angels'] = df_round.apply(add_investor,investor=angels, axis=1 )
    return df_round

def get_ipo_targets():
    df_ipo = get_ipos_df()
    df_ipo_target = df_ipo.set_index("object_id")
    df_ipo_target.columns = ["target"]
    df_ipo_target = df_ipo_target.apply(lambda x: 1, axis =1)
    df_ipo_target.name = "ipo_target"
    return df_ipo_target

def get_acq_targets():
    df_acq = get_acquisitions_df()
    df_acq_target = df_acq[['acquired_object_id',"acquisition_id"]].set_index("acquired_object_id")
    df_acq_target.columns = ["target"]
    df_acq_target = df_acq_target.apply(lambda x: 1, axis =1)
    df_acq_target.name = "acq_target"
    return df_acq_target

def get_closed_targets():
    df_object = get_objects_df()
    df_closed = df_object[df_object['status'] == "closed"]
    df_closed = df_closed.set_index('id')
    df_closed = df_closed.apply( lambda x: 0, axis =1)
    return df_closed

def get_pos_targets():
    df_acq = get_acquisitions_df()
    df_ipo = get_ipos_df()
    left = df_ipo.set_index("object_id")
    right = df_acq[['acquired_object_id',"acquisition_id"]].set_index("acquired_object_id")
    target_objects = left.join(right, rsuffix="_object", how="outer")
    target = target_objects['ipo_id'].apply(lambda x: 1)
    return target

def insert_target(x,index=None,col=None, **kwargs):
    if x[col] in index:
        return 1
    else:
        return 0

def get_all_targets(X):
    X = X.set_index("id")
    target = get_pos_targets()
    index = set(list(target.index))
    X = X.reset_index()
    X['target'] = pd.Series(np.random.randn(len(X)), index=X.index)
    X['target'] = X.apply(insert_target,index=index,col="id", axis=1)
    
    #target = X.pop("target")
    #target.name = "target"
    #return target
    return X

def get_acquisitions_df():
    df_acquisitions = pd.read_csv('../data/cb_acquisitions.csv', parse_dates=['acquired_at'])
    keep_acquisitions = ['acquisition_id', 'acquired_object_id', 'price_amount', 'acquired_at']
    df_acquisitions = df_acquisitions[keep_acquisitions]
    return df_acquisitions

def get_degrees_df():
    pass
def get_funding_rounds_df():
    df = pd.read_csv('../data/cb_funding_rounds.csv', parse_dates=['funded_at'])
    keep = ['funding_round_id','object_id', 'funded_at', 'funding_round_code','raised_amount_usd','participants','is_first_round']
    df = df[keep]
    return df

def get_funds_df():
    df = pd.read_csv('../data/cb_funds.csv', parse_dates=['funded_at'])
    keep = ['fund_id', 'object_id', 'name', 'funded_at']
    df = df[keep]
    return df

def get_investor_type(x):
    let = x['investor_object_id'][0]
    if let == 'c':
        return 'company'
    elif let == 'p':
        return 'angel'
    elif let == 'f':
        return 'fund'
    else:
        return 'other'

def get_investments_df():
    df = pd.read_csv('../data/cb_investments.csv', parse_dates=['created_at', 'updated_at'])
    keep = ['funding_round_id', 'funded_object_id', 'investor_object_id']
    
    df = df[keep]
    df['investor_type'] = df.apply(get_investor_type, axis=1 )
    return df


def get_ipos_df():
    df = pd.read_csv('../data/cb_ipos.csv', parse_dates=['public_at'])
    keep = ['ipo_id', 'object_id']
    df = df[keep]
    return df


def get_milestones_df():
    df = pd.read_csv('../data/cb_milestones.csv',parse_dates=['milestone_at'])
    keep = ['object_id', 'milestone_at', 'description']
    df = df[keep]
    return df

def get_objects_df():
    df = pd.read_csv('../data/cb_objects.csv',parse_dates=['founded_at'])
    keep = ['id','normalized_name','category_code','status','founded_at','domain',
 'overview','tag_list','country_code','state_code', 'city', 'region']

    df = df[keep]
    new_df = df[['founded_at', 'category_code', 'id']]
    new_df['years_since'] = new_df['founded_at'].apply(lambda x: (pd.Timestamp('01-01-2014') - x).days)
    index = list(new_df['category_code'].value_counts(1)[:20].index)
    #new_df[index] = pd.get_dummies(df['category_code'])[index]
    new_df.drop(columns=['category_code','founded_at'], inplace=True)
    return new_df

def get_offices_df():
    df = pd.read_csv('../data/cb_offices.csv')
    keep = []
    df = df[keep]
    return df


def get_people_df():
    df = pd.read_csv('../data/cb_people.csv')
    keep = []
    df = df[keep]
    return df

def get_relationships_df():
    df = pd.read_csv('../data/cb_relationships.csv', parse_dates=['start_at','end_at'])
    keep = ['person_object_id','relationship_object_id','start_at','end_at','is_past','title']
    df = df[keep]
    return df


