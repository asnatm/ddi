import os
import random
from src.data_readers.d2d_releases_reader import d2d_releases_reader
import pandas as pd

from src.data_readers.dataset_dates import d2d_versions_metadata
from src.drug_classification.single_drug_features_creator import DrugBank_drug_feature_creator
from sqlalchemy import create_engine

from src.main.data_migration.table_names import drugBank_schema, get_DB_connection, combine_table_name_version

os.chdir('..\\..\\..')

version = "5.1.5"
force_read_file=False

def insert_drugBank_id_to_DB(version,force_read_file):
    d2d_releases_r1 = d2d_releases_reader(force_read_file=force_read_file)
    drug_reader = d2d_releases_r1.read_release(version)
    drug_data = pd.DataFrame(list(drug_reader.drug_id_to_cat.keys())).reset_index()
    drug_data.columns = ['index', 'drugBank_id']
    drug_features = DrugBank_drug_feature_creator(drug_reader)
    data_to_insert = [
        drug_features.get_drug_list(),
        drug_features.create_cat_features(return_sparse=False),
        drug_features.create_weight_features(),
        drug_features.create_tax_features(),
        drug_features.create_smiles_features(),
        drug_features.get_target_features(return_sparse=False),
        drug_features.get_enzymes_features(return_sparse=False),
        drug_features.get_carriers_features(return_sparse=False),
        drug_features.get_transporter_features(return_sparse=False),
        drug_features.get_associated_conditions(return_sparse=False),#TODO: has no version, ths is collected periodically
        drug_features.get_types(return_sparse=False),
        drug_features.get_groups(return_sparse=False),
    ]
    for level in range(1,6):
        data_to_insert.append(drug_features.create_ATC_features(level=level, return_sparse=False,return_description=False))
    for level in range(1, 5): # last level has no description
        data_to_insert.append(drug_features.create_ATC_features(level=level, return_sparse=False,return_description=True))

    db_engine = get_DB_connection()
    con = db_engine.connect()
    for df in data_to_insert:
        print('inserting',df.index.name,'version',version)
        df.to_sql(combine_table_name_version(df.index.name, version), con=con, schema=drugBank_schema, index=False,
                  if_exists='replace')
    con.close()
    db_engine.dispose()



# random.seed(30)
# for version_details in d2d_versions_metadata:
#     try:
#         version = version_details['VERSION']
#         insert_drugBank_id_to_DB(version)
#     except:
#         print('#########################################cannot read',version_details)
# d2d_releases_r1 = d2d_releases_reader(force_read_file=False)
# drug_reader = d2d_releases_r1.read_release("5.1.5")
insert_drugBank_id_to_DB('5.1.5',force_read_file=True)
