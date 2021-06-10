import os
import pickle
import random
import pandas as pd
from pandas import HDFStore

from src.data_path import unlabled_path, unlabled_text_path, modalities_dict_file, hdf_path
from src.drug_classification.get_drugBank_features_from_DB import GetDrugBankFeaturesFromDB
from src.drug_classification.get_dense_vectors_features_from_DB import GetDenseVectorsFeaturesFromDB
from src.main.data_migration.table_names import *

os.chdir('..\\..\\..')

random.seed(30)
version = "5.1.5"
t = GetDrugBankFeaturesFromDB()
ans,ans_modalities = t.combine_features([category_table,
                                         ATC_table_1,ATC_table_2,ATC_table_3,ATC_table_4,ATC_table_5,
                                            ATC_table_1_description,ATC_table_2_description,ATC_table_3_description,ATC_table_4_description,
                                         enzyme_table,carrier_table,target_table,transporter_table,associated_condition_table,group_table,type_table],
                         dense_table_name_list=[mol_weight_table,smiles_table,tax_table],version=version,add_counts_to_sparse=True)

ans_text_features = t.combine_string_features([category_table,ATC_table_1_description,ATC_table_2_description,enzyme_table,
                                               carrier_table,associated_condition_table], version=version)
for c in [ans_text_features.name]:
    assert c not in ans.columns
ans_modalities['text'] = [ans_text_features.name]
ans = ans.join(ans_text_features,how='left')
#assert ans.isna().any().sum()==1,"only smiles should have nan"
#ans = ans.fillna(0) #fill na for dense features.


mol2vec = GetDenseVectorsFeaturesFromDB()

print('adding mol2vec')
mol2vec_features = mol2vec.get_mol2Vec_features()
for c in mol2vec_features:
    assert c not in ans.columns
ans_modalities['mol2vec'] = list(mol2vec_features.columns)
ans = ans.join(mol2vec_features,how='left',on='Smiles')



print('adding AMFP')
AMFP_features = mol2vec.get_AMFP_features(version=version)
for c in AMFP_features:
    assert c not in ans.columns
ans_modalities['AMFP'] = list(AMFP_features.columns)
ans = ans.join(AMFP_features,how='left')
ans = ans.fillna(ans.mean())

#ans.to_csv(unlabled_path,index=True)

ans_modalities = pd.DataFrame({'modality':[x for x in sorted(ans_modalities.keys()) for y in ans_modalities[x]],'feature':[y for x in sorted(ans_modalities.keys()) for y in ans_modalities[x]]})
#ans_modalities.to_csv(modalities_dict_file,index=False)

for c in ans_modalities['feature'].values:
    assert c in ans.columns, "canot find column in df "+c

store = pd.HDFStore('output\data\modalities_dict.h5')
store['df'] = ans
store['modalities'] = ans_modalities
store.close()