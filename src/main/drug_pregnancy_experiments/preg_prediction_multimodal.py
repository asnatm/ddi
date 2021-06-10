from pandas import HDFStore
from sklearn.cluster import  AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import os
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import  StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler

from src.data_path import *
from src.data_readers.tagged_preg_reader import tagged_data_reader
from src.drug_classification.supervised_dimensionality_reduction import supervised_dim_reduction, convert_to_one_hot, \
    getPCA, extract_text_features, get_col_clusters
from src.main.drug_pregnancy_experiments.pregnancy_drug_experment import  transform_labels, preg_scorer
os.chdir('..\\..\\..')
repeats = 5

class filter_cols():
    def __init__(self,cols):
        self.cols=cols

    def fit(self,X,y=None):
        pass
    def transform(self,X,y=None):
        ans = X[self.cols]
        return ans.values
    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X,y)

#Read data
store = HDFStore('output\data\modalities_dict.h5')
X_unlabled = store['df']
X_unlabled = X_unlabled.loc[~X_unlabled.index.duplicated(keep='first')]#four drugs appear twice....

modalities = store['modalities']
print('done reading from disk')

X_unlabled,modalities = convert_to_one_hot(X_unlabled,modalities,'Taxonomy')

# for m in ['ATC_Level_2_description']: #After running, this is the only one needed. x for x in modalities.modality.unique()
#     try:
#         features = modalities.loc[modalities.modality.isin([m]), 'feature']
#         X_unlabled_pca = getPCA(X_unlabled,features,m)
#         for c in X_unlabled_pca.columns:
#             modalities = modalities.append({'modality':m+'_pca','feature':c},ignore_index=True)
#         X_unlabled=X_unlabled.join(X_unlabled_pca,how='left')
#     except:
#         print('cannot pca',m)

# for m in ['Taxonomy']: #After running, this is the only one needed. x for x in modalities.modality.unique()
#     try:
#         data = X_unlabled[modalities[modalities.modality.isin([m])].feature]
#         X_m_cat = supervised_dim_reduction(data, X_unlabled[modalities[modalities.modality.isin(['Category'])].feature].drop('Number of Category', axis=1))
#         X_m_cat.columns =[m + '_cat_' + str(x) for x in X_m_cat.columns]
#         X_m_cat.index = X_unlabled.index
#         for c in X_m_cat.columns:
#             modalities = modalities.append({'modality':m+'_cat','feature':c},ignore_index=True)
#         X_unlabled=X_unlabled.join(X_m_cat, how='left')
#     except:
#         print('cannot dim red',m)

if True: ##After running, Not helping
    X_unlabled,modalities = extract_text_features(X_unlabled,modalities)
    print('done processing text')


if True: #cluster category using text
    a = get_col_clusters(X_unlabled,modalities)
    X_unlabled=X_unlabled.join(a,how='left')
    for c in a.columns:
        modalities = modalities.append({'modality': 'category_cluster', 'feature': c}, ignore_index=True)

print("done preproc")

# ###### CV exp
preg_tagged_data_reader = tagged_data_reader()
X_cv = X_unlabled.join(preg_tagged_data_reader.read_all(remove_disputed=True,read_smc=True,read_Eltonsy_et_al=True,read_safeFetus=False),how='inner')
y_cv = X_cv['preg_class']
y_cv = pd.Series(transform_labels(y_cv),index=y_cv.index,name='tag')
del X_cv['preg_class']

# X_cv.to_csv(X_data_path, index=False)
#y_cv.to_csv(y_data_path, index=True)

params ={'max_features': 32, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 387,'n_jobs':6}

def cv_exp(modalities_to_experemint,results_collector):
    for rep in range(repeats):
        kf = StratifiedKFold(n_splits=10, random_state=rep, shuffle=True)
        for cv_i, (train_index, test_index) in enumerate(kf.split(X_cv, y_cv)):
            X_train, X_test = X_cv.iloc[train_index], X_cv.iloc[test_index]
            y_train, y_test = y_cv.iloc[train_index], y_cv.iloc[test_index]
            list_pipe = []
            for m in modalities_to_experemint:
                for clf_name, clf in [('Extra trees', ExtraTreesClassifier(n_jobs=4))]:
                    try:
                        text_pipe = Pipeline(
                            [('select', filter_cols(modalities.loc[modalities.modality.isin(m), 'feature'])),
                             ('pred', clf)])
                        text_pipe.fit(X_train, y_train)
                        text_preds = text_pipe.predict_proba(X_test)[:, 1]
                        list_pipe.append((m, text_pipe))
                        results_collector.add_score(m, clf_name, y_test, text_preds, rep, cv_i, 0)
                    except:
                        print('cannot process',clf_name,m)
        print(rep)
    return results_collector

base_mods = [[x]for x in modalities.modality.unique()]
base_mods.remove(['text'])#unstuctred
base_mods.remove(['Smiles'])#unstuctred
base_mods.remove(['ATC_Level_1_description'])#Double
base_mods.remove(['ATC_Level_2_description'])#Double
base_mods.remove(['ATC_Level_3_description'])#Double
base_mods.remove(['ATC_Level_4_description'])#Double
#base_mods = [[x] for x in ['Category', 'Carrier', 'Group', 'mol2vec_cat','ATC_Level_3','Taxonomy']]

modalities_to_evaluate = base_mods
results_collector = preg_scorer()
best_auc = 0.5
current_base_mod = []
while True:
    print('Current auc', best_auc, 'base mods:', current_base_mod)
    modalities_to_evaluate = [current_base_mod + x for x in base_mods if x[0] not in current_base_mod]
    print('working on',modalities_to_evaluate)
    results_collector = cv_exp(modalities_to_evaluate,results_collector)
    current_base_mod, new_auc=results_collector.get_best_mod()
    if new_auc-best_auc<=0.0:
        break
    best_auc=new_auc
print(results_collector.as_df().groupby('Name').mean().sort_values(by='AUC',))
best_mods = results_collector.get_best_mod()[0]




##Cross DB
preg_tagged_data_reader = tagged_data_reader()
train_set = X_unlabled.join(preg_tagged_data_reader.read_all(remove_disputed=True,read_who=True,read_smc=False,read_Eltonsy_et_al=True,read_safeFetus=False),how='inner')
y_train = train_set['preg_class']
y_train = pd.Series(transform_labels(y_train),index=y_train.index,name='tag')
X_train = train_set
del X_train['preg_class']

test_set = X_unlabled.join(preg_tagged_data_reader.read_all(remove_disputed=True,read_who=False,read_smc=True,read_Eltonsy_et_al=False,read_safeFetus=False),how='inner')
print('Removing drug appearing in train ans test both:',len(set(test_set.index) & set(X_train.index)))

test_set = test_set.loc[~test_set.index.isin(train_set.index)]
y_test = test_set['preg_class']
y_test = pd.Series(transform_labels(y_test),index=y_test.index,name='tag')
X_test = test_set
del X_test['preg_class']


results_collector_db = preg_scorer()
ExtraTrees_preds=None

for m in [best_mods]:
    for clf_name, clf in [('Extra trees', ExtraTreesClassifier(**params)) ]:
        for rep in range(repeats):
            try:
                ExtraTrees_pipe = Pipeline(
                    [('select', filter_cols(modalities.loc[modalities.modality.isin(m),'feature'])),
                     ('pred', ExtraTreesClassifier())])
                ExtraTrees_pipe.fit(X_train, y_train)
                ExtraTrees_preds = ExtraTrees_pipe.predict_proba(X_test)[:, 1]
                results_collector_db.add_score(m, clf_name, y_test, ExtraTrees_preds, rep, 0, 0)
            except:
                pass
print(results_collector_db.as_df().groupby('Name').mean().sort_values(by='AUC'))
#results_cross_db.to_csv('output\data\exp_results_cross_db.csv')
print('Number of drugs in train but not in test:',len(set(X_train.index) - set(X_test.index)))
print('Number of drugs in test but not in train:',len(set(X_test.index) - set(X_train.index)))
print('Number of drugs in test but not in both:',len(set(X_test.index) & set(X_train.index)))




##Corona
preg_tagged_data_reader = tagged_data_reader()
train_set = X_unlabled.join(preg_tagged_data_reader.read_all(remove_disputed=True,read_smc=True,read_Eltonsy_et_al=False,read_safeFetus=False),how='inner')
y_train = train_set['preg_class']
y_train = pd.Series(transform_labels(y_train),index=y_train.index,name='tag')
X_train = train_set
del X_train['preg_class']


corona_set = X_unlabled.join(preg_tagged_data_reader.read_SMC_corona(),how='inner')
y_corona = corona_set['preg_class']
y_corona = pd.Series(transform_labels(y_corona),index=y_corona.index,name='tag')
X_corona = corona_set
del X_corona['preg_class']

results_collector_corona = preg_scorer()
for m in [best_mods]:
    for i in range(10):
        try:
            ExtraTrees_pipe = Pipeline(
                [('select', filter_cols(modalities.loc[modalities.modality.isin(m),'feature'])),
                 ('pred', ExtraTreesClassifier(**params))])
            ExtraTrees_pipe.fit(X_train, y_train)
            corona_preds = ExtraTrees_pipe.predict_proba(X_corona)[:, 1]
            results_collector_corona.add_score(m,'Trees', y_corona, corona_preds, 0, 0, 0)
        except:
            pass
print(results_collector_corona.as_df().groupby('Name').mean().sort_values(by='AUC').AUC)
print(results_collector_corona.as_df().groupby('Name').mean().sort_values(by='AUC').Kappa)



res = X_unlabled.join(preg_tagged_data_reader.read_SMC_corona(),how='inner').join(preg_tagged_data_reader.read_who(),how='left',rsuffix='_who').join(preg_tagged_data_reader.read_smc(),how='left',rsuffix='_smc')
ExtraTrees_pipe = Pipeline(
    [('select', filter_cols(modalities.loc[modalities.modality.isin(best_mods), 'feature'])),
     ('pred', ExtraTreesClassifier(**params))])
ExtraTrees_pipe.fit(X_train, y_train)
corona_preds = ExtraTrees_pipe.predict_proba(X_corona)[:, 1]
res['pred'] = corona_preds
print(res[['pred','preg_class','preg_class_who','preg_class_smc']].sort_values(by=['pred']))



import shap
clf = ExtraTreesClassifier()
select_mod_shap = modalities.loc[modalities.modality.isin(best_mods), 'feature']
clf.fit(X_train[select_mod_shap].astype(float),y_train)
data_shap = X_corona[select_mod_shap].astype(float)
explainer = shap.TreeExplainer(clf,data_shap)
shap_values = explainer.shap_values(data_shap)
shap_class_idx=1
shap.summary_plot(shap_values[shap_class_idx],data_shap,max_display=20,show=True,
                   color_bar=True,class_names=['Low Risk','High Risk'])
shap_values_df = pd.DataFrame(data_shap)
drug_id='DB09065'
curr_shap_value = shap_values_df.loc[shap_values_df.index==drug_id,:]#explainer.shap_values(curr_drug_features)
shap.force_plot(explainer.expected_value[shap_class_idx],
                curr_shap_value.values, data_shap.columns, matplotlib=True, show=True,out_names=['Probability higher risk'],figsize=(20,4)#
                )#out_names=['Lower Risk','Higher Risk']

from src.data_readers.d2d_releases_reader import d2d_releases_reader
d2d_releases_r1 = d2d_releases_reader()
drug_reader, drug_preproc1 = d2d_releases_r1.read_and_preproc_release("5.1.5")
drug_names=pd.DataFrame({'drugBank_id':list(drug_reader.drug_id_to_name.keys()),'Drug name':list(drug_reader.drug_id_to_name.values())})
drug_names=drug_names.set_index('drugBank_id')
