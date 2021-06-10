import pandas as pd

inter = pd.read_excel('interactions.xlsx',header=None)
inter.columns = ['drug1','drug2']
inter.drug1 = inter.drug1.str.strip()
inter.drug2 = inter.drug2.str.strip()
drugs=pd.DataFrame(covid_results_df['Drug name'])
drugs.columns=['drugBank_name']
inter = pd.merge(inter,drugs,left_on='drug1',right_on='drugBank_name',how='left')
inter = pd.merge(inter,drugs,left_on='drug2',right_on='drugBank_name',how='left')
#interactions = inter[~((inter.drugBank_name.isna()))]
interactions = inter[~((inter.drugBank_name_x.isna()) & (inter.drugBank_name_y.isna()))]
