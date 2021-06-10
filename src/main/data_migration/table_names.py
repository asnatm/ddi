from sqlalchemy import create_engine


def get_DB_connection():
    engine = create_engine(
        'postgresql+psycopg2://drugsmaster:pass2DRUGS!@132.72.64.83/postgres')  # pip install psycopg2
    return engine


def combine_table_name_version(table_name,version):
    return table_name + "_" + version

################# Mol2Vec
mol2VecSchema = 'Mol2Vec'
mol2VecTable='smiles2vec_300dim'


#################AMFP
AMFP_schema = 'AMFP'
AMFP_table = 'AMFP'

###############DrugBank
#Key
drugBank_id = 'drugBank_id'
#schema
drugBank_schema = 'DrugBank'

#Tables
drug_table = 'drug'
category_table = "Category"
ATC_table_1 = 'ATC_Level_1'
ATC_table_2 = 'ATC_Level_2'
ATC_table_3 = 'ATC_Level_3'
ATC_table_4 = 'ATC_Level_4'
ATC_table_5 = 'ATC_Level_5'

atc_description_suffix = '_description'

ATC_table_1_description = 'ATC_Level_1' + atc_description_suffix
ATC_table_2_description = 'ATC_Level_2' + atc_description_suffix
ATC_table_3_description = 'ATC_Level_3'+ atc_description_suffix
ATC_table_4_description = 'ATC_Level_4' + atc_description_suffix
#ATC_table_5_description = 'ATC_Level_5' + atc_description_suffix # has no description

tax_table = 'Taxonomy'
mol_weight_table = 'Molecular_weight'
smiles_table='Smiles'
target_table = 'Target'
enzyme_table = 'Enzyme'
carrier_table = 'Carrier'
transporter_table = 'Transporter'
group_table = 'Group'
type_table = 'Type'
associated_condition_table = 'Associated_condition'

all_sparse_features = [category_table,
                       ATC_table_1,
                       ATC_table_2,
                       ATC_table_3,
                       ATC_table_4,
                       ATC_table_5,
                       target_table,
                       enzyme_table,
                       carrier_table,
                       transporter_table,
                       associated_condition_table,
                       ]
all_non_sparse_features = [mol_weight_table]