import os
from matplotlib import pyplot as plt
import shap

from src.data_readers.d2d_releases_reader import d2d_releases_reader
from src.data_readers.tagged_nCoV_reader import nCoV_tagged_data_reader

from src.data_path import *
from src.data_readers.tagged_preg_reader import tagged_data_reader
from src.main.drug_pregnancy_experiments.pregnancy_drug_experment import getExtraTreesOptimized, transform_labels, \
    getVoting, run_CV_experement, run_cross_DB_experiment

os.chdir('..\\..\\..')



X_unlabled = load_data()
X_unlabled=X_unlabled.set_index('drugBank_id')

# pca = PCA(n_components=85)
# ids = X_unlabled['drugBank_id']
# del X_unlabled['drugBank_id']
# X_unlabled_pca = pca.fit_transform(StandardScaler().fit_transform(X_unlabled))
# X_unlabled_pca = pd.DataFrame(X_unlabled_pca)
# X_unlabled_pca['drugBank_id'] =ids
#
# X_unlabled = pd.concat([X_unlabled,X_unlabled_pca],axis=1)



print('done reading from disk')

# ###### CV exp
preg_tagged_data_reader = tagged_data_reader()
X_cv = X_unlabled.join(preg_tagged_data_reader.read_all(True),how='inner')
y_cv = X_cv['preg_class']
y_cv = pd.Series(transform_labels(y_cv))

#del X_cv['drugBank_id']
del X_cv['preg_class']
X_cv.to_csv(X_data_path, index=False)
y_cv.to_csv(y_data_path, index=False)
print('done merging')

#X_cv  = X_cv[[x for x in X_cv.columns if 'emb_' not in x]]

results = run_CV_experement(X_cv, y_cv, repeats=10)
results.to_csv('output\data\exp_results.csv')
print(results.groupby('Name').mean())
results.groupby('Name').mean().to_csv('output\data\exp_summarized_results.csv')


#
#
# ##### crosss db exp
# preg_tagged_data_reader = tagged_data_reader()
# train_set = X_unlabled.join(preg_tagged_data_reader.read_who(remove_disputed=False),how='inner')
# #del train_set['drugBank_id']
# y_train = train_set['preg_class']
# X_train = train_set
# del X_train['preg_class']
#
# test_set = X_unlabled.join(preg_tagged_data_reader.read_smc(remove_disputed=False),how='inner')
# #del test_set['drugBank_id']
# y_test = test_set['preg_class']
# X_test = test_set
# del X_test['preg_class']
#
# results_cross_db = run_cross_DB_experiment(X_train,X_test,y_train,y_test)
# print(results_cross_db.groupby('Name').mean())
# results_cross_db.to_csv('output\data\exp_results_cross_db.csv')
#
#
#
#
#
# ##### SHAP
# preg_tagged_data_reader = tagged_data_reader()
# X = X_unlabled.join(preg_tagged_data_reader.read_all(),how='inner')
# y = X['preg_class']
# tagged_drugs_ids = X.index#['drugBank_id']
# #del X['drugBank_id']
# del X['preg_class']
#
#
# clf = getExtraTreesOptimized()
# clf.fit(X.astype(float), transform_labels(y))
#
# clf_voting = getVoting()
# clf_voting.fit(X.astype(float), transform_labels(y))
#
# data_shap = X_unlabled
# #data_shap['drugBank_id'] = X_unlabled['drugBank_id']
# data_shap = data_shap.loc[(data_shap.index.isin(tagged_drugs_ids)) | (data_shap.index.isin(nCoV_tagged_data_reader.drug_id_to_name.keys()))]
# #data_shap = data_shap.reset_index(drop=True)
# #drug_ids = data_shap['drugBank_id']
# #del data_shap['drugBank_id']
#
# data_shap = data_shap.astype(float)
#
# #explainer = shap.KernelExplainer(clf.predict,data_shap)
# explainer = shap.TreeExplainer(clf,data_shap)
# shap_values = explainer.shap_values(data_shap)
# shap_class_idx=1
#
#
# shap_values_df = pd.DataFrame(shap_values[shap_class_idx])
# shap_values_df.index=data_shap.index
#
#
# plt.clf()
# shap.summary_plot(shap_values[shap_class_idx],data_shap,max_display=20,show=False,
#                   color_bar=True,class_names=['Low Risk','High Risk'])
# img = plt.gcf()
# img.show()
# img.savefig("output\\SHAP\\shap_model.pdf",bbox_inches='tight',pad_inches =0.2) #dpi=150,
# #shap.summary_plot(shap_values, X, plot_type="bar",max_display=10)
#
#
# shap.initjs()
# covid_results = {}
# covid_results['drugBank_id']=[]
# covid_results['prediction']=[]
# covid_results['probability_ExtraTreesClassifier_safe']=[]
# covid_results['probability_voting_safe']=[]
#
#
# for drug_id in sorted(nCoV_tagged_data_reader.drug_id_to_name.keys()):
#         #drug_idx = drug_ids[drug_ids==drug_id].index.values[0]
#         curr_drug_features = data_shap.loc[data_shap.index==drug_id,:]
#         curr_shap_value = shap_values_df.loc[shap_values_df.index==drug_id,:]#explainer.shap_values(curr_drug_features)
#         p = shap.force_plot(explainer.expected_value[shap_class_idx],
#                         curr_shap_value.values, data_shap.columns, matplotlib=True, show=False,out_names=['Probability higher risk'],figsize=(20,4)#
#                         )#out_names=['Lower Risk','Higher Risk']
#         p.show()
#         p.savefig("output\\SHAP\\" + drug_id + '.pdf'#,bbox_inches='tight'
#                   )#format='eps', , dpi=150
#         pred_clf = clf.predict_proba(curr_drug_features)[0, 1]
#         pred_clf_voting = clf_voting.predict_proba(curr_drug_features)[0, 1]
#         current_pred= 'Safe' if pred_clf <= 0.34 else 'Limited'
#         covid_results['drugBank_id'].append(drug_id)
#         covid_results['prediction'].append(current_pred)
#         covid_results['probability_ExtraTreesClassifier_safe'].append(pred_clf)
#         covid_results['probability_voting_safe'].append(pred_clf_voting)
#
#
# d2d_releases_r1 = d2d_releases_reader()
# drug_reader, drug_preproc1 = d2d_releases_r1.read_and_preproc_release("5.1.6")
#
# covid_results_df = pd.DataFrame(covid_results)
# covid_results_df = covid_results_df.set_index('drugBank_id')
# covid_results_df = covid_results_df.join(preg_tagged_data_reader.read_who(remove_disputed=False),how='left')
# covid_results_df=covid_results_df.rename(columns={'preg_class':'Classification by Polifka et al.'})
# covid_results_df = covid_results_df.join(preg_tagged_data_reader.read_smc(remove_disputed=False),how='left')
# covid_results_df=covid_results_df.rename(columns={'preg_class':'Classifiaction by TIS Zerifin'})
# covid_results_df['drug_name'] = [drug_reader.drug_id_to_name[x] for x in covid_results_df.index]
# covid_results_df = covid_results_df.rename(columns={'drug_name': 'Drug name',
#     'prediction':'Prediction','probability_ExtraTreesClassifier_safe':'Probability higher risk by ExtraTrees model',
#                                                     'probability_voting_safe':'Probability higher risk by voting of models'},index = {'drugBank_id':'DrugBank id'})
# #covid_results_df = covid_results_df[['Drug name','DrugBank id','Prediction','Probability higher risk']]
# covid_results_df=covid_results_df.sort_values(by=['Drug name'])
# covid_results_df = covid_results_df[['Drug name', 'Prediction', 'Probability higher risk by ExtraTrees model','Probability higher risk by voting of models',
#        'Classification by Polifka et al.', 'Classifiaction by TIS Zerifin'
#        ]]
# del covid_results_df['Prediction']
# covid_results_df.to_csv('output\\data\\covid_drugs_predictions.csv',index=True)
#
# joined_dbs = preg_tagged_data_reader.read_smc(remove_disputed=False).join(preg_tagged_data_reader.read_who(remove_disputed=False),how='inner',lsuffix='_asaf', rsuffix='_polifica')
# drug_names=pd.DataFrame({'drugBank_id':list(drug_reader.drug_id_to_name.keys()),'Drug name':list(drug_reader.drug_id_to_name.values())})
# joined_dbs = pd.merge(joined_dbs,drug_names,how='left',left_index=True,right_index=True)
# joined_dbs.to_csv('output\\data\\dbCompare.csv')
#
#
#
#
# plt.clf()
# shap.summary_plot(shap_values[shap_class_idx],data_shap,max_display=10,show=False,
#                   color_bar=True,class_names=['Low Risk','High Risk'],plot_type="bar")
# img = plt.gcf()
# img.show()