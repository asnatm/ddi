import os
import pandas as pd
os.chdir('..\\..\\..')

tagged_drugs_path = os.path.join('pickles', 'data', 'preg',
                                 'WHO essential med classification.xlsx')  # r'C:\Users\Administrator\PycharmProjects\DDI_SMC\data\WHO essential med classification.xlsx'
ans = pd.read_excel(tagged_drugs_path)
ans = ans.set_index('generic_drug_name')
res = {}
res['generic_drug_name'] = []
res['separated_name'] = []
res['comment']=[]
for d in ans.index.values:
    try:
        cmnt =d.split('(')[1].replace(')','')
    except:
        cmnt=''
    #print(''.join([i if ord(i) >= 128 else '' for i in d]))
    for d_part in d.split('(')[0].split('+'):
        res['generic_drug_name'].append(d)
        res['separated_name'].append(d_part.strip())
        res['comment'].append(cmnt)

        #print(d,':',d_part)

res_df = pd.DataFrame(res).set_index('generic_drug_name')
ans.join(res_df,how='left').to_csv(os.path.join('pickles', 'data', 'preg',
                                 'WHO essential med classification processed.csv') )