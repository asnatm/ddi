import os
import pandas as pd
from src.data_readers.d2d_releases_reader import d2d_releases_reader
from src.data_readers.tagged_nCoV_reader import nCoV_tagged_data_reader

os.chdir('..\\..\\..')

base = r'''\begin{figure}[htbp]
\makebox[\textwidth]{\includegraphics[width=0.9\paperwidth]{Figures/%s.pdf}}
\caption{{\bf Feature importance using SHAP for %s}}
\label{fig:shap_drugs_%s}
\end{figure}'''

base+='\n'

d2d_releases_r1 = d2d_releases_reader()
drug_reader, drug_preproc1 = d2d_releases_r1.read_and_preproc_release("5.1.5", force_read_file=False)

db_id = sorted(nCoV_tagged_data_reader.drug_id_to_name.keys())
d_name = [drug_reader.drug_id_to_name[x] for x in db_id]

drugs_data = pd.DataFrame({'drugBank_id':db_id,'drug name':d_name})
drugs_data = drugs_data.sort_values(by=['drug name'])

ans = str()
for i,row in enumerate(drugs_data.values):
    #drug_name = drug_reader.drug_id_to_name[drug_id]
    drug_id,drug_name = row
    ans += str(base % (drug_id,drug_name,i))
    ans+='\n'
print(ans)
text_file = open(r"output\data\figures.txt", "w")
n = text_file.write(ans)
text_file.close()
