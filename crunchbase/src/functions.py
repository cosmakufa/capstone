import pandas as pd
import numpy as np 
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
def insert_target(x,index=None, **kwargs):
    if x['index'] in index:
        return 1
    else:
        return 0

def make_target(X):
    X = X.copy()
    df_ipo = pd.read_csv('data/cb_ipos.csv')
    df_acquisitions = pd.read_csv('data/cb_acquisitions.csv')
    left = df_ipo[['object_id','ipo_id']].set_index("object_id")
    right = df_acquisitions[['acquired_object_id',"acquisition_id"]].set_index("acquired_object_id")
    ipo_acquisition = left.join(right, rsuffix="_object", how="outer")
    target = ipo_acquisition['ipo_id'].apply(lambda x: 1)
    index = set(list(target.index))
    X = X.reset_index()
    X['target'] = pd.Series(np.random.randn(len(X)), index=X.index)
    X['target'] = X.apply(insert_target,index=index, axis=1)
    target.name = "target"
    target = X.pop("target")
    return target

def feature_engineer():
    df_objects = pd.read_csv('data/cb_objects.csv')
    df_ipo = pd.read_csv('data/cb_ipos.csv')
    df_acquisitions = pd.read_csv('data/cb_acquisitions.csv')
    keep =['id', 'parent_id', 
        'normalized_name', 'category_code',  'founded_at',
        'domain', 'twitter_username',
        'short_description', 'description',
        'overview', 'tag_list', 'country_code', 'state_code', 'city', 'region',
        'first_investment_at', 'last_investment_at', 'investment_rounds',
        'invested_companies', 'first_funding_at', 'last_funding_at',
        'funding_rounds', 'funding_total_usd', 'first_milestone_at',
        'last_milestone_at', 'milestones', 'relationships'
        ]
    removed = [ 'entity_type', 'entity_id', 'name', 'permalink', 'status', 'closed_at', 'homepage_url',  'logo_url', 'logo_width', 'logo_height','created_by', 'created_at', 'updated_at']
    keeps =[  'parent_id', 
       'normalized_name', 'category_code',  'founded_at',
        'domain', 'twitter_username',
        'short_description', 'description',
       'overview', 'tag_list', 'country_code', 'state_code', 'city', 'region','status',
       'first_investment_at', 'last_investment_at', 'investment_rounds',
       'invested_companies', 'first_funding_at', 'last_funding_at',
       'funding_rounds', 'funding_total_usd', 'first_milestone_at',
       'last_milestone_at', 'milestones', 'relationships',
       ]
    df_objects = df_objects[(df_objects['status'].isin(['acquired','closed', 'ipo']))]
    df_objects = df_objects[keep]
    df_joined = df_objects.set_index("id").join(df_ipo.set_index("object_id"), rsuffix="_ipo").join(df_acquisitions.set_index("acquired_object_id"),  rsuffix="_acq")
    df_num = df_joined[keep]._get_numeric_data()
    df_num = df_num.fillna(df_num.mean())
    return df_num

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

def confusion_matrix_percent(target, y_hat):
    vc_Structure = np.array([[-1,0],[0,100]])
    result = confusion_matrix(target,y_hat) / len(target) * 100
    return {"cost_benefit": result * vc_Structure, "confu_matrix": result, "expected_value": np.sum(result * vc_Structure)}

def get_current_model(X, y):
    clf = RandomForestClassifier(max_depth=5, random_state=0, class_weight='balanced')
    model = clf.fit(X.values, y.values)
    return model
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def current_flow():
    X = feature_engineer()
    target = make_target(X)
    X_train, X_test, y_train, y_test = split_data(X, target)
    model = get_current_model(X_train,y_train)
    y_hat_p= np.array(list(list(zip(* model.predict_proba(X_test.values)))[0]))
    return X_test, y_test , y_hat_p
