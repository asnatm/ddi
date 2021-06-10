import pandas as pd
from src.main.data_migration.table_names import get_DB_connection, mol2VecSchema, mol2VecTable, AMFP_table, \
    combine_table_name_version, drugBank_id, AMFP_schema


class GetDenseVectorsFeaturesFromDB():

    def __init__(self):
        pass

    def read_table(self,table_name,index_col,schema):
        db_engine = get_DB_connection()
        con = db_engine.connect()
        ans =  pd.read_sql_table(table_name=table_name,con=con,index_col=index_col,schema=schema)
        con.close()
        db_engine.dispose()
        return ans

    def get_mol2Vec_features(self):
        table =  self.read_table(mol2VecTable,index_col='SMILES',schema=mol2VecSchema)
        return table

    def get_AMFP_features(self, version):
        table =  self.read_table(combine_table_name_version(AMFP_table, version),index_col=drugBank_id,schema=AMFP_schema)
        return table
