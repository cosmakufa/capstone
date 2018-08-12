import pandas as pd


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

def get_investments_df():
    df = pd.read_csv('../data/cb_investments.csv', parse_dates=['created_at', 'updated_at'])
    keep = ['funding_round_id', 'funded_object_id', 'investor_object_id']
    df = df[keep]
    return df


def get_ipos_df():
    df = pd.read_csv('../data/cb_ipos.csv', parse_dates=['public_at'])
    keep = ['ipo_id', 'object_id']
    df = df[keep]
    return df


def get_milestones_df():
    df = pd.read_csv('../data/cb_milestones.csv')
    keep = ['object_id', 'milestone_at', 'description']
    df = df[keep]
    return df

def get_objects_df():
    df = pd.read_csv('../data/cb_objects.csv')
    keep = ['id','normalized_name','category_code','status','founded_at','domain',
 'overview','tag_list','country_code','state_code', 'city', 'region']

    df = df[keep]
    return df

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
