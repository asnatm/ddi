import os
import pickle
import pandas as pd

from src.data_readers.d2d_DAL import drug_data_reader
from src.drug_name_resolver.drugBank_name_resolver import drugBank_name_resolver


def set_pandas_options() -> None:
    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 199
    pd.options.display.width = None
    # pd.options.display.precision = 2  # set as needed
set_pandas_options()


# combines the manually tagged drugs with who tagged drugs.

who_data = pd.read_excel(os.path.join('data', 'WHO essential med classification.xlsx'))
tagged_drugs = pd.read_excel(os.path.join('data', 'taggedDrugs.xls'))


drugBank_file =os.path.join('data','drugbank','5.1.4', 'drugbank_all_full_database.xml.zip')
drugBank_pickle_file =os.path.join('data','drugbank','5.1.4', 'drugBankPickle.p')
if not os.path.isfile(drugBank_pickle_file ) :
    drugbank = drug_data_reader(drugBank_file )
    drugbank.read_data_from_file()
    f=open(drugBank_pickle_file,'wb')
    pickle.dump(drugbank, f,pickle.HIGHEST_PROTOCOL)
    f.close()
else:
    f = open(drugBank_pickle_file,'rb')
    drugbank= pickle.load(f)
    f.close()
name_resolved = drugBank_name_resolver(drugbank)
tagged_drugs=name_resolved.resolve_names(tagged_drugs)
print(tagged_drugs.describe())


combined_tags=pd.merge(tagged_drugs,who_data[who_data.drugBank_id.notnull()],how='left',on='generic_drug_name')
non_merged = set(who_data.generic_drug_name) - set(tagged_drugs.generic_drug_name)
combined_tags.to_excel(os.path.join('data', 'taggedDrugs_w_who.xls'),index=False)
