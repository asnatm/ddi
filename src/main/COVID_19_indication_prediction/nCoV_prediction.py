from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import shap
from sklearn.feature_selection import SelectKBest,  mutual_info_classif
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold, StratifiedKFold
import os
from src.data_readers.d2d_releases_reader import d2d_releases_reader
from src.data_readers.tagged_nCoV_reader import nCoV_tagged_data_reader
from src.drug_classification.single_drug_features_creator import DrugBank_drug_feature_creator
import random

os.chdir('..\\..\\..')


new_version="5.1.4"


target_att= 'In_Final_List'
random.seed(30)
d2d_releases_r1 = d2d_releases_reader()
drug_reader, drug_preproc1 = d2d_releases_r1.read_and_preproc_release(new_version)


#test_tuples=set(test_tuples)
number_of_drugs = len(drug_reader.all_drugs)


drug_data=pd.DataFrame(list(drug_reader.drug_id_to_cat.keys())).reset_index()
drug_data.columns=['index', 'drugBank_id']

preg_tagged_data_reader = nCoV_tagged_data_reader()
tagged_data = preg_tagged_data_reader.read_zhou_et_al_tagged()
drug_data=drug_data.merge(tagged_data, how='left', on='drugBank_id')
tagged_data = drug_data.dropna(axis=0, subset=[target_att])
assert len(tagged_data.drugBank_id)==tagged_data.drugBank_id.nunique()


drug_features = DrugBank_drug_feature_creator(drug_reader)
cat_features_dataframe = drug_features.create_cat_features()
atc_features_dataframe = drug_features.create_ATC_features(level=2)
zhau_features_dataframe = drug_features.get_zahu_et_al_features()
genname_features_dataframe = drug_features.get_target_features()
indication_features_dataframe = drug_features.get_associated_conditions()


zhau_features_labeled = pd.merge(zhau_features_dataframe,tagged_data.drop(columns=['index'],axis=0),on='drugBank_id',how='inner')
zhau_features_unlabled =zhau_features_dataframe

cat_features_dataframe_lableled = pd.merge(cat_features_dataframe,tagged_data.drop(columns=['index'],axis=0),on='drugBank_id',how='inner')
cat_features_dataframe_unlabeled = cat_features_dataframe#[~cat_features_dataframe.drugBank_id.isin(cat_features_dataframe_lableled.drugBank_id.values)]

atc_features_dataframe_lableled = pd.merge(atc_features_dataframe,tagged_data.drop(columns=['index'],axis=0),on='drugBank_id',how='inner')
atc_features_dataframe_unlabeled = atc_features_dataframe#[~atc_features_dataframe.drugBank_id.isin(atc_features_dataframe_lableled.drugBank_id.values)]

genname_features_dataframe_lableled = pd.merge(genname_features_dataframe,tagged_data.drop(columns=['index'],axis=0),on='drugBank_id',how='inner')
genname_features_dataframe_unlabeled = genname_features_dataframe#[~atc_features_dataframe.drugBank_id.isin(atc_features_dataframe_lableled.drugBank_id.values)]


indication_features_dataframe_lableled = pd.merge(indication_features_dataframe,tagged_data.drop(columns=['index'],axis=0),on='drugBank_id',how='inner')
indication_features_dataframe_unlabeled = indication_features_dataframe#[~atc_features_dataframe.drugBank_id.isin(atc_features_dataframe_lableled.drugBank_id.values)]


# pca = PCA(n_components=5)
# X_interaction_pca= pca.fit_transform(X_interaction)
# X_interaction_kbest= SelectKBest(k=1).fit_transform(X_interaction, y)

# pca = PCA(n_components=10)
#X_ATC_pca= pca.fit_transform(preprocessing.scale(X_ATC))
# kbest_cat = SelectKBest(k=50)
# X_ATC_kbest= kbest_cat.fit_transform(X_ATC, y)

#y = cat_features_dataframe_lableled[target_att]
y = zhau_features_labeled[target_att]
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

#pca = PCA(n_components=300)
#X_interaction_pca= pca.fit_transform(X_cat_kbest_train)
# X_gen_kbest = SelectKBest(k=10).fit_transform(X_gennames, y)

# X = X_atc_kbest_labeled
# X_unlabled=X_atc_kbest_unlabled
# Drugbank_ids_prediction = atc_features_dataframe_unlabeled.drugBank_id

# X = X_cat_kbest_labeled
# X_unlabled=X_cat_kbest_unlabled
# Drugbank_ids_prediction = cat_features_dataframe_unlabeled.drugBank_id

xs = [zhau_features_labeled.drop(columns=[target_att], axis=0),
       cat_features_dataframe_lableled.drop(columns=[target_att], axis=0),
       atc_features_dataframe_lableled.drop(columns=[target_att], axis=0),
       genname_features_dataframe_lableled.drop(columns=[target_att], axis=0),
       indication_features_dataframe_lableled.drop(columns=[target_att], axis=0),
      ]
xs2 = []
for df in xs:
    xs2.append(df.set_index(['drugBank_id']))
X=None
for df in xs2:
    if X is None:
        X = df
    else:
        X=X.join(df,how='left',rsuffix='r')
X = X.fillna(0)

# xs = [zhau_features_unlabled,
#       #cat_features_dataframe_unlabeled,
#       #atc_features_dataframe_unlabeled,
#       #
#      genname_features_dataframe_unlabeled,
#       #indication_features_dataframe_unlabeled
#       ]
# xs2 = []
# for df in xs:
#     xs2.append(df.set_index(['drugBank_id']))
# X_unlabled=None
# for df in xs2:
#     if X_unlabled is None:
#         X_unlabled = df
#     else:
#         X_unlabled=X_unlabled.join(df,how='left',rsuffix='r')
#
# X_unlabled = X_unlabled.fillna(0)

print('working on k best')
print('orig num features:',len(X.columns))
kbest = SelectKBest(mutual_info_classif,k=50)
kbest.fit(X,y)
selected_columns = [X.columns[x] for x in kbest.get_support(True)]
print('selected',selected_columns)
X = X[selected_columns]

# #
# kbest_cat = SelectKBest(k=300)
# X_cat_kbest_labeled = kbest_cat.fit_transform(cat_features_dataframe_lableled.drop(columns=['drugBank_id',target_att], axis=0), y) #Manual Search, PubMed. https://dev.drugbankplus.com/guides/fields/drugs?_ga=2.251280646.1017776629.1574595982-1885976375.1573671598
# X_cat_kbest_unlabled= kbest_cat.transform(cat_features_dataframe_unlabeled.drop(columns=['drugBank_id'],axis=0))
# ##adding names
# cols = kbest_cat.get_support(indices=True)
# features = cat_features_dataframe_lableled.drop(columns=['drugBank_id',target_att], axis=0).columns
# X_cat_kbest_labeled= pd.DataFrame(X_cat_kbest_labeled,columns=[features[x] for x in cols])
# X_cat_kbest_unlabled= pd.DataFrame(X_cat_kbest_unlabled,columns=[features[x] for x in cols])
#
#
#
# kbest_atc = SelectKBest(k='all')
# X_atc_kbest_labeled = kbest_atc.fit_transform(atc_features_dataframe_lableled.drop(columns=['drugBank_id',target_att], axis=0), y)
# X_atc_kbest_unlabled= kbest_atc.transform(atc_features_dataframe_unlabeled.drop(columns=['drugBank_id'],axis=0))
#
# cols = kbest_atc.get_support(indices=True)
# features = atc_features_dataframe_lableled.drop(columns=['drugBank_id',target_att], axis=0).columns
# X_atc_kbest_labeled= pd.DataFrame(X_atc_kbest_labeled,columns=[features[x] for x in cols])
# X_atc_kbest_unlabled= pd.DataFrame(X_atc_kbest_unlabled,columns=[features[x] for x in cols])
#
#
# kbest_zhaou = SelectKBest(k='all')
# X_zahou_kbest_labeled = kbest_zhaou.fit_transform(zhau_features_labeled.drop(columns=['drugBank_id',target_att], axis=0), y)
# X_zahou_kbest_unlabled = kbest_zhaou.transform(zhau_features_unlabled.drop(columns=['drugBank_id'],axis=0))
#
# cols = kbest_zhaou.get_support(indices=True)
# features = zhau_features_labeled.drop(columns=['drugBank_id',target_att], axis=0).columns
# X_zahou_kbest_labeled= pd.DataFrame(X_zahou_kbest_labeled,columns=[features[x] for x in cols])
# X_zahou_kbest_unlabled= pd.DataFrame(X_zahou_kbest_unlabled,columns=[features[x] for x in cols])



def return_ranks(array):
    temp = np.array(array).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks

all_scores=[]
pred_df = X.copy()
pred_columns=set()
for rep in range(10):
    kf = StratifiedKFold(n_splits=3,random_state=rep,shuffle=True)
    #kf.get_n_splits(X,y)
    for cv_i ,(train_index, test_index) in enumerate(kf.split(X,y)):
        #clf = GradientBoostingClassifier(n_estimators=1000)
        # clf = XGBClassifier(n_estimators=800,learning_rate=0.5,max_features=1)
        #clf = LogisticRegression()
        clf = RandomForestClassifier(n_estimators=1000)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        column_name='pred_'+str(rep)+'_'+str(cv_i)
        pred_columns.add(column_name)
        curr_pred = [x[1] for x in clf.predict_proba(X)]
        pred_df[column_name+'pred'] = curr_pred  # zero means safe. 1 means avoid.
        pred_df[column_name] = return_ranks(curr_pred)  # zero means safe. 1 means avoid.
        pred_df[column_name+'_is_test'] = [1 if x in test_index else 0 for x in range(len(X))]
        all_scores.append(roc_auc_score(y_test, [x[1] for x in clf.predict_proba(X_test)]))
#print('single model score', roc_auc_score(le.transform(y_test), [x[1] for x in clf.predict_proba(X_test)]))
all_scores=np.array(all_scores)
print(rep+1,"repeats auc: %0.6f (+/- %0.2f)" % (all_scores.mean(), all_scores.std() * 2))
pred_df['pred']=pred_df[pred_columns].mean(axis=1)


pred_df = pred_df.join(pd.Series(drug_reader.drug_id_to_name,name='names'))
pred_df = pred_df.join(tagged_data.set_index('drugBank_id'),how='left')
pred_df = pred_df.sort_values(by=['pred'],ascending=False)
print(pred_df.head(20))
pred_df.to_csv('temp2.csv')
#pred_df.sort_values(by=['preg_pred'],ascending=False).merge(tagged_data,how='left',on="drugBank_id").to_excel('temp.xls')

# cat_features_dataframe = drug_features.create_multilabeled_feature({x:[y for y in drug_reader.drug_id_to_cat[x]] for x in drug_reader.drug_id_to_cat})
# test_tagged1= cat_features_dataframe.merge(preg_tagged_data_reader.read_smc(),on='drugBank_id',how='inner')
# test_tagged1.to_csv('test.xls')
# test_tagged1[target_att] = le.transform(test_tagged1[target_att])
# corr1 = test_tagged1.corr()
# corr1=corr1[target_att]
#
# test_tagged2=cat_features_dataframe.merge(preg_tagged_data_reader.read_who(),on='drugBank_id',how='inner')
# test_tagged2.to_csv('test2.xls')
# test_tagged2[target_att] = le.transform(test_tagged2[target_att])
# corr2 = test_tagged2.corr()
# corr2=corr2[target_att]
#
# corr=pd.DataFrame(corr1).join(pd.DataFrame(corr2),rsuffix='who')
# corr.to_excel('corr.xls')
#
# count=test_tagged1.sum(axis=0).astype(bool)*1
# count2=test_tagged2.sum(axis=0).astype(bool)*1
#
# # clf = RandomForestClassifier()
# clf.fit(X,y)
# #
explainer = shap.TreeExplainer(clf)
# explainer = shap.LinearExplainer(clf,X)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X,max_display=30)
shap.summary_plot(shap_values, X, plot_type="bar",max_display=10)
#

file_name =  r"Combiation therapy_A61K.xlsx"
import pandas as pd
df = pd.read_excel(io=file_name)
print(df.head(5))  # print first 5 rows of the dataframe
