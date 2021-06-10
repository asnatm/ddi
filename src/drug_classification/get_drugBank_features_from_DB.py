import pandas as pd
from src.main.data_migration.table_names import *


class GetDrugBankFeaturesFromDB():

    def __init__(self):
        pass
    def read_table(self,table_name):
        db_engine = get_DB_connection()
        con = db_engine.connect()
        ans = pd.read_sql_table(table_name=table_name,con=con,index_col=drugBank_id,schema=drugBank_schema)
        con.close()
        db_engine.dispose()
        return ans



    def get_feature(self,feature_name,version,return_sparse=True):
        table =  self.read_table(combine_table_name_version(feature_name,version))

        if return_sparse:
            assert len(table.columns)==1,"supports only one column"
            column = table.columns[0]
            table=table.reset_index()
            constant_value_tmp_name = 'count_constant'
            table[constant_value_tmp_name]=True
            table = table.pivot_table(index='drugBank_id', columns=column, values=constant_value_tmp_name, aggfunc='any',
                           fill_value=False).astype(bool)
            return table
        else:
            return table

    #combines drugs, contains all drugs which appeared in the version
    #missing values for sparse features are filled with False
    #missing values for dense features are not filled.
    def combine_features(self,sparse_table_name_list,dense_table_name_list,version,add_name_prefix_to_sparse=True,add_counts_to_sparse=False):

        bool_columns = list()
        float_columns = list()
        ans_modalities = {}
        ans = self.get_feature(drug_table,version,return_sparse=False)
        for t in sparse_table_name_list:
            name = t.replace("_"," ")
            print('merging table',name)
            current_dataframe = self.get_feature(t,version,return_sparse=True)
            if add_name_prefix_to_sparse:
                current_dataframe = current_dataframe.add_prefix(name+": ")
            bool_columns.extend(current_dataframe.columns)
            if add_counts_to_sparse:
                col_name='Number of '+name
                current_dataframe.loc[:,col_name] = current_dataframe.sum(axis=1)
                float_columns.append(col_name)

            sums = current_dataframe.sum()
            limit = 2
            sums = sums[sums >= limit]
            current_dataframe = current_dataframe.loc[:, sums.index]  # filtering out categories which appeared only for a single drug

            ans_modalities[t] = list(current_dataframe.columns)
            for c in current_dataframe.columns:
                assert c not in ans.columns, 'column in two modaliaties '+str(c)
            ans = ans.join(current_dataframe,how='left')


        print('filling NA')
        #ans = ans.fillna(value={**{x:False for x in bool_columns},**{x:0 for x in float_columns}})
        ans = ans.fillna(0)
        print('casting')

        for f in [x for x in ans.columns]:
            if f in bool_columns:
                ans[f] = ans[f].astype(bool)
            else:
                assert f in float_columns
                ans[f] = ans[f].astype(float)

        for t in dense_table_name_list:
            print('adding',t)
            current_dataframe = self.get_feature(t,version,return_sparse=False)
            ans_modalities[t] = list(current_dataframe.columns)
            for c in current_dataframe.columns:
                assert c not in ans.columns, 'column in two modaliaties '+str(c)

            ans = ans.join(current_dataframe,how='left')


        return ans,ans_modalities

    def combine_string_features(self,sparse_table_name_list,version):
        ans = self.get_feature(drug_table,version,return_sparse=False)
        ans['text']=""
        for t in sparse_table_name_list:
            name = t.replace("_"," ")
            print('merging table',name)
            current_dataframe = self.get_feature(t,version,return_sparse=False)
            assert len(current_dataframe.columns)==1,"only one sparse feature is supported"
            current_dataframe.columns = ['text']
            ans = ans.append(current_dataframe)
        ans = ans.groupby(ans.index)['text'].apply(lambda x: "%s " % ' '.join(x))
        ans=ans.str.strip()
        return ans

# version = "5.1.6"
# t = GetDrugBankFeaturesFromDB()
# ans = t.combine_features([category_table,ATC_table_1],dense_table_name_list=[mol_weight_table],version=version)
