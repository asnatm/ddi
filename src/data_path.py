import pandas as pd

X_data_path = 'output\data\X.csv'
y_data_path = 'output\data\y.csv'
unlabled_path =r'output\data\unlabled_path.csv'
modalities_dict_file = r'output\data\modalities_dict.p'
hdf_path  = r'output\data\modalities_dict.p'
unlabled_text_path =r'output\data\unlabled_text_path.csv'
tagged_drugs_ids_path = r'output\data\tagged_drugbank_id.csv'
unlabled_drugs_ids_path = r'output\data\unlabled_drugbank_id.csv'


def load_data():

    X_unlabled = pd.read_csv(unlabled_path)

    return X_unlabled
