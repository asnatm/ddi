
import os
import numpy as np
import pandas as pd

from src.main.data_migration.table_names import *





class DrugBank_drug_feature_creator():

    def __init__(self, drug_reader):
        self.drug_reader=drug_reader
        #sorted list of drugs.
        #self.drugs=drugs

    def get_associated_conditions(self, return_sparse=True):
        tagged_drugs_path = os.path.join('pickles', 'data', 'indications',
                                         'drugs_indications_df.csv')  # r'C:\Users\Administrator\PycharmProjects\DDI_SMC\data\WHO essential med classification.xlsx'
        tagged_drugs = pd.read_csv(tagged_drugs_path)

        tagged_drugs['drugBank_id']=tagged_drugs['drug'].astype(str)
        del tagged_drugs['drug']
        #del tagged_drugs['ind_id']
        del tagged_drugs['approved']
        tagged_drugs = tagged_drugs.dropna(subset=['drugBank_id'])
        ans = {}
        for d in tagged_drugs.groupby('drugBank_id').ind_lbl.apply(list).reset_index().values:
            ans[d[0]]=d[1]
        for d in self.drug_reader.all_drugs:
            if d not in ans:
                ans[d]=[]
        if return_sparse:
            ans = self.create_multilabeled_feature(ans)
        else:
            ans = self.create_non_sparse_result(ans,'Associated_Conditions')

        # cols = []
        # for c in tagged_drugs.columns:
        #     if c!= 'Number of indications' and c!='drugBank_id':
        #         cols.append('Indication: '+c)
        #     else:
        #         cols.append(c)
        # tagged_drugs.columns=cols
        ans.index.name = associated_condition_table
        return ans

    # def get_zahu_et_al_features(self):
    #     tagged_drugs_path = os.path.join('pickles', 'data', 'nCoV',
    #                                      'Drug_screening.xlsx')  # r'C:\Users\Administrator\PycharmProjects\DDI_SMC\data\WHO essential med classification.xlsx'
    #     tagged_drugs = pd.read_excel(tagged_drugs_path)
    #     tagged_drugs['drugBank_id']=tagged_drugs['drugBank_id'].astype(str)
    #     tagged_drugs = tagged_drugs.dropna(subset=['drugBank_id'])
    #     tagged_drugs = tagged_drugs[['drugBank_id',
    #                                  'Pan_ZSCORE', 'SARS_ZSCORE', 'MERS_ZSCORE', 'IBV_ZSCORE', 'MHV_ZSCORE']]
    #
    #     return tagged_drugs

    def create_multilabeled_feature(self,input_id_to_strings):
        # recieve dict with id as key and list of strings to transform
        #make sure the list of drugs and the keys are the same sets
        #assert set(self.drugs)==set(input_id_to_strings.keys()),'list of sorted drugs is not equal to the set of keys of drugs. '+str(len(self.drugs)) + " " + str(len(set(input_id_to_strings.keys()))) + str(set(self.drugs).symmetric_difference(set(input_id_to_strings.keys())))
        unique_strings = set()
        for d in input_id_to_strings:
            unique_strings = unique_strings | set(input_id_to_strings[d])
        unique_strings = set(unique_strings)
        #print('strings:',unique_strings)
        string_features = {x: list() for x in unique_strings}
        drug_list=sorted(input_id_to_strings)
        #print('drug list',drug_list)
        for d in drug_list:
            for string in string_features:
                string_features[string].append(0)  # add all zeros. later change non-zero
            for string in input_id_to_strings[d]:
                string_features[string][-1] = string_features[string][-1] + 1  # count non zero

        for f in string_features:
            string_features[f] = np.array(string_features[f], dtype=bool) #convert to bool.

        ans_dataframe = pd.DataFrame.from_dict(
            {'drugBank_id': drug_list, **string_features})
        ans_dataframe['drugBank_id']=ans_dataframe['drugBank_id'].astype(str)

        return self.remove_forbidden_carcters_from_col_names(ans_dataframe)

    def remove_forbidden_carcters_from_col_names(self,df):
        df.columns = [x.replace('<','-').replace('[','(').replace(']',')') for x in df.columns]
        return df

    def align_test_features_w_trian(self, cat_features_dataframe_train, cat_features_dataframe_test):
        columns_train=cat_features_dataframe_train.columns
        columns_test = cat_features_dataframe_test.columns
        print('len intersecting cols:',len(set(columns_train)&set(columns_test)))
        print('len redundant cols:', len(set(columns_test) - set(columns_train)))
        print('len missing cols:', len(set(columns_train) - set(columns_test)))
        ans = pd.DataFrame(cat_features_dataframe_test)
        ans=ans[[x for x in columns_train if x in columns_test]]
        for x in [c for c in columns_train if c not in columns_test]:
            ans[x] = 0
        return ans

    def create_cat_features(self,return_sparse=True):
        ans = {x:[y for y in self.drug_reader.drug_id_to_cat[x]] for x in self.drug_reader.drug_id_to_cat}
        if return_sparse:
            ans =  self.create_multilabeled_feature(ans)
        else:
            ans = self.create_non_sparse_result(ans,'Category')
        ans.index.name = category_table
        return ans

    def create_non_sparse_result(self, ans,field_name):
        drug_ids = [str(x) for x in sorted(ans.keys()) for y in ans[x]]
        values = [str(y) for x in sorted(ans.keys()) for y in ans[x]]
        ans = pd.DataFrame({'drugBank_id': drug_ids,field_name : values})
        return ans

    def create_interaction_feature(self):
        pass
    #
    #
    # validaition_instance_features=None
    # target_att=None
    # amfp_params = {'mul_emb_size' : 512, 'dropout':0.4, 'epochs':6, 'batch_size':1024, 'learning_rate':0.01,'propagation_factor':0.4}
    # #amfp_params = {'mul_emb_size': 256, 'dropout': 0.3, 'epochs': 6, 'batch_size': 256, 'learning_rate': 0.01,              'propagation_factor': None}

    # nn = drugs_nn_predictor(m_train,test_tuples,validation_tuples=validaition_instance_features,validation_target=target_att, name='AMFP', **amfp_params)
    # predictors.append(nn)
    # nn.fit()
    # predictions = nn.predict()
    # add_predictor_evaluation(preds=predictions,name=nn.name)
    #
    # for i in range(len(overall_auc)):
    #     print(f'Name: {overall_auc[i][1]}, auc: {overall_auc[i][0]}, map: {overall_ap[i][0]}, aupr: {overall_aupr[i]}, AP@k: {overall_k_precision[i][0][0]},{overall_k_precision[i][0][1]},{overall_k_precision[i][0][2]},{overall_k_precision[i][0][3]},{overall_k_precision[i][0][4]}')


    # input_node_a = Input(shape=(1,), name='b')
    # emb_mult = nn.mult_dense(input_node_a)
    # model_emb = Model(inputs=[input_node_a], outputs=emb_mult)  # fixed_input
    # model_emb.compile(optimizer=Adam())  # binary_crossentropy
    # emb_mlp1 = nn.mlp(input_node_a)
    # model_mlp = Model(inputs=[input_node_a], outputs=emb_mlp1)  # fixed_input
    # model_mlp.compile(optimizer=Adam())  # binary_crossentropy
    # mult_pred=model_emb.predict(tagged_data['index'])
    # mlp_pred = model_mlp.predict(tagged_data['index'])
    #
    # X_mult = np.array([x[0] for x in mult_pred])
    # X_mlp = np.array([x[0] for x in mlp_pred])
    # X_interaction=np.concatenate((X_mult, X_mlp), axis=1)



    def get_target_features(self,return_sparse=True):
        ans = {x:[y for y in self.drug_reader.drug_id_to_targets[x]] for x in self.drug_reader.drug_id_to_targets}
        if return_sparse:
            ans = self.create_multilabeled_feature(ans)
        else:
            ans = self.create_non_sparse_result(ans,'Target')
        # newCols = ['Target: ' + x for x in ans.columns if x!='drugBank_id']
        # newCols = ['drugBank_id'] + newCols
        # ans.columns=newCols
        ans.index.name=target_table
        return ans

    def get_enzymes_features(self,return_sparse=True):
        ans =  {x: [y for y in self.drug_reader.drug_id_to_enzymes[x]] for x in self.drug_reader.drug_id_to_enzymes}
        if return_sparse:
            ans = self.create_multilabeled_feature(ans)
        else:
            ans= self.create_non_sparse_result(ans,'Enzyme')

        # newCols = ['Enzyme: ' + x for x in ans.columns if x!='drugBank_id']
        # newCols = ['drugBank_id'] + newCols
        # ans.columns = newCols
        ans.index.name = enzyme_table
        return ans

    def get_drug_list(self):

        ans = pd.DataFrame({'drugBank_id':self.drug_reader.all_drugs})
        ans.index.name = drug_table

        return ans

    def get_carriers_features(self,return_sparse=True):
        ans = {x: [y for y in self.drug_reader.drug_id_to_carriers[x]] for x in self.drug_reader.drug_id_to_carriers}
        if return_sparse:
            ans = self.create_multilabeled_feature(ans)
        else:
            ans= self.create_non_sparse_result(ans,'Carrier')

        # newCols = ['Carrier: ' + x for x in ans.columns if  x != 'drugBank_id']
        # newCols = ['drugBank_id'] + newCols
        # ans.columns = newCols
        ans.index.name = carrier_table

        return ans

    def get_transporter_features(self,return_sparse=True):
        ans = {x: [y for y in self.drug_reader.drug_id_to_transporters[x]] for x in self.drug_reader.drug_id_to_transporters}
        if return_sparse:
            ans = self.create_multilabeled_feature(ans)
        else:
            ans= self.create_non_sparse_result(ans,'Transporter')

        # newCols = ['Transporter: ' + x for x in ans.columns if  x != 'drugBank_id']
        # newCols = ['drugBank_id'] + newCols
        # ans.columns = newCols
        ans.index.name = transporter_table

        return ans
        # gen_feature_dataframe = drug_features.create_multilabeled_feature({x:drug_reader.drug_id_to_genname[x] for x in drug_reader.drug_id_to_genname if x in tagged_data['drugBank_id'].to_numpy()})
        # del gen_feature_dataframe['drugBank_id']
        # X_gennames = gen_feature_dataframe.to_numpy()
    def create_ATC_features(self, level=2,return_sparse=True,return_description=False):
        assert level in {1,2,3,4,5}
        drug_atcs = self.drug_reader.drug_id_to_ATC
        ans=None
        index_name=None
        if level == 1:
            ans = {x: list(set([y[0] for y in drug_atcs[x]])) for x in drug_atcs}#just first letter
            index_name = ATC_table_1
            # atc_translation_1 = {x: "ATC: " + self.drug_reader.atc_to_text[x].title() + " ("+x+")" for x in atc_first_letter_dataframe.columns if x!='drugBank_id'}
            # atc_first_letter_dataframe.rename(columns=atc_translation_1, inplace=True)
            #ans = atc_first_letter_dataframe

        elif level ==2:
            ans = {x: list(set([y[:3] for y in drug_atcs[x]])) for x in drug_atcs} #second level is two letter starting at second
            index_name = ATC_table_2
            # atc_translation_2 = {x: "ATC: " + self.drug_reader.atc_to_text[x].title() + " ("+x+")" for x in atc_second_letter_dataframe.columns if x!='drugBank_id'}
            # atc_second_letter_dataframe.rename(columns=atc_translation_2,inplace=True)
            # ans = pd.merge(ans,atc_second_letter_dataframe,on='drugBank_id',how='inner')
        elif level ==3:

            ans = {x: list(set([y[:4] for y in drug_atcs[x]])) for x in drug_atcs} #second level is two letter starting at second
            index_name = ATC_table_3
            # atc_translation_2 = {x: "ATC: " + self.drug_reader.atc_to_text[x].title() + " ("+x+")" for x in atc_third_letter_dataframe.columns if x!='drugBank_id'}
            # atc_third_letter_dataframe.rename(columns=atc_translation_2,inplace=True)
            # ans = pd.merge(ans,atc_third_letter_dataframe,on='drugBank_id',how='inner')
        elif level ==4:
            ans = {x: list(set([y[:5] for y in drug_atcs[x]])) for x in drug_atcs} #second level is two letter starting at second
            index_name = ATC_table_4
        elif level ==5:
            ans = {x: list(set([y[:7] for y in drug_atcs[x]])) for x in drug_atcs} #second level is two letter starting at second
            index_name = ATC_table_5

        if return_description:
            ans = {x:[self.drug_reader.atc_to_text[y] for y in ans[x]] for x in ans}
            index_name=index_name+atc_description_suffix

        if return_sparse:
            ans = self.create_multilabeled_feature(ans)
        else:
            ans = self.create_non_sparse_result(ans, 'ATC '+str(level))
        ans.index.name = index_name

        return ans

    def create_weight_features(self):
        drugs = list(self.drug_reader.drug_id_to_weight.keys())
        ans = pd.DataFrame({'drugBank_id':drugs,'Molecular weight':[self.drug_reader.drug_id_to_weight[x] for x in drugs]})
        ans["Molecular weight"] = ans["Molecular weight"].astype(float)
        # #print(ans.weight.describe())
        # feature_name = 'Molecular weight > 10000Da'
        # ans[feature_name] = 0
        # t=float(10000)#ans.weight.median()
        # ans.loc[ans.weight<t,feature_name] = 0
        # ans.loc[ans.weight >= t, feature_name] = 1
        # assert not ans.weight.isna().any()
        # #ans.loc[ans.weight.isna(), 'weight_class'] = 0
        # del ans['weight']
        # ans[feature_name] = ans[feature_name].astype(bool)
        ans.index.name = mol_weight_table
        return ans

    def create_tax_features(self):
        drugs = list(self.drug_reader.drug_id_to_tax_description.keys())
        ans = pd.DataFrame({'drugBank_id':drugs,
                            'Kingdom':[self.drug_reader.drug_id_to_tax_kingdom[x] for x in drugs],
                            'Super Class': [self.drug_reader.drug_id_to_tax_superclass[x] for x in drugs],
                            'Class': [self.drug_reader.drug_id_to_tax_class[x] for x in drugs],
                            'Sub Class': [self.drug_reader.drug_id_to_tax_subclass[x] for x in drugs],
                            'Direct Parent': [self.drug_reader.drug_id_to_tax_direct_parent[x] for x in drugs],
                            }) #TODO: add more fields from taxonomy, add the description
        ans.index.name = tax_table
        return ans

    def create_smiles_features(self):
        drugs = list(self.drug_reader.drug_id_to_smiles.keys())
        ans = pd.DataFrame(
            {'drugBank_id': drugs, 'Smiles': [self.drug_reader.drug_id_to_smiles[x] for x in drugs]})
        ans.index.name = smiles_table
        return ans

    def get_types(self, return_sparse):
        ans = {x: [self.drug_reader.drug_to_type[x]] for x in
               self.drug_reader.drug_to_type}
        if return_sparse:
            ans = self.create_multilabeled_feature(ans)
        else:
            ans = self.create_non_sparse_result(ans, 'Type')
        ans.index.name = type_table
        return ans

    def get_groups(self, return_sparse):
        ans = {x: [y for y in self.drug_reader.drug_id_to_groups[x]] for x in
               self.drug_reader.drug_id_to_groups}
        if return_sparse:
            ans = self.create_multilabeled_feature(ans)
        else:
            ans = self.create_non_sparse_result(ans, 'Group')
        ans.index.name = group_table
        return ans


