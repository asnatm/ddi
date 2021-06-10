import os
import random

from tpot import TPOTClassifier
from sklearn.model_selection import  StratifiedKFold
import numpy as np
from src.data_path import load_data
from src.data_readers.tagged_preg_reader import tagged_data_reader
from src.main.drug_pregnancy_experiments.pregnancy_drug_experment import transform_labels

os.chdir('..\\..\\..')


X_unlabled = load_data()

preg_tagged_data_reader = tagged_data_reader()
X = X_unlabled.merge(preg_tagged_data_reader.read_all(),on='drugBank_id',how='inner')
y = X['preg_class']
del X['drugBank_id']
del X['preg_class']
y = np.array(transform_labels(y))

# X.to_csv(X_data_path, index=False)
# pd.DataFrame(y).to_csv(y_data_path, index=False)

kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
tpot = TPOTClassifier(generations=5, population_size=5,
                      cv=kf,scoring='roc_auc',max_time_mins=60*3,verbosity=2,config_dict='TPOT light') #verbosity=1,
#config_dict='TPOT sparse' 'TPOT MDR' 'TPOT light'
print(tpot)
tpot.fit(X, y)
tpot.export('tpot_model'+str(random.uniform(1.5, 1.9))+'.py')
#
#
# all_auc = []
# for i in range(10):
#     kf = StratifiedKFold(n_splits=10, random_state=0,  shuffle=True)
#     for cv_i, (train_index, test_index) in enumerate(kf.split(X, y)):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         tpot.fitted_pipeline_.fit(X_train,y_train)
#         try:
#             all_auc.append(roc_auc_score(y_test, [x[1] for x in tpot.predict_proba(X_test)]))
#         except:
#             all_auc.append(roc_auc_score(y_test, [x for x in tpot.predict(X_test)]))
# print("100 repeats auc: %0.3f (+/- %0.2f)" % (np.array(all_auc).mean(), np.array(all_auc).std()))
