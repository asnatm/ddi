import pandas as pd

#takes drug names as input and returns the drugbank ids for them.
class drugBank_name_resolver():

    def __init__(self,drugBank):
        """
        :param drugBank:
        :type drugBank: drug_data_reader
        """

        self.drugBank=drugBank
        cols_name=[(self.drugBank.drug_id_to_name[x].lower(),x) for x in self.drugBank.drug_id_to_name]
        self.name_to_id=pd.DataFrame.from_dict({'name':[x[0] for x in cols_name],'id_name':[x[1] for x in cols_name]})
        cols_syn=[(y.lower(),x) for x in self.drugBank.drug_to_synonyms for y in self.drugBank.drug_to_synonyms[x]]
        self.syn_to_id= pd.DataFrame.from_dict({'syn':[x[0] for x in cols_syn],'id_syn':[x[1] for x in cols_syn]})

    def resolve_names(self,drug_names):
        drug_names=pd.DataFrame(drug_names)
        drug_names.generic_drug_name=drug_names.generic_drug_name.str.lower()
        resolved_names=pd.merge(drug_names,self.name_to_id,how='left',left_on='generic_drug_name',right_on='name')
        resolved_names = pd.merge(resolved_names, self.syn_to_id, how='left', left_on='generic_drug_name', right_on='syn')

        syn = self.syn_to_id[self.syn_to_id.syn.notnull()]
        print(resolved_names.head())
        # resolved_names['drugBank_id'] = resolved_names.apply(
        #     lambda row: row['id_name'] if not np.isnan(row['id_name']) else row['id_syn'],
        #     axis=1
        # )
        resolved_names['drugBank_id']=resolved_names['id_name']
        resolved_names.loc[(pd.isnull(resolved_names.drugBank_id)), 'drugBank_id'] = resolved_names.id_syn

        print('duplicate syns')
        print(syn[self.syn_to_id.syn.duplicated(keep=False)])
        # print(resolved_names.describe())
        resolved_names.drop(['id_syn', 'id_name','name','syn'], axis=1,inplace=True)
        return resolved_names

