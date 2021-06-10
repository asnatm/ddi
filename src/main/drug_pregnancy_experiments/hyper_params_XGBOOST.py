import os
import warnings
from sklearn.metrics import roc_auc_score
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
from src.data_path import load_data
from src.data_readers.tagged_preg_reader import tagged_data_reader
from src.main.drug_pregnancy_experiments.pregnancy_drug_experment import transform_labels

os.chdir('..\\..\\..')
large_number = 1


def score(params):
    print("Training with params : ")
    print(params)
    # for p in ['n_estimators','num_k_best']:
    #     params[p] = int(params[p])
    stat = STATUS_OK
    curr_score=0

    all_auc = []
    try:
        model = xgb.XGBClassifier(**params)
        kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
        for cv_i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            try:
                test_preds = model.predict_proba(X_test)
                test_preds = [x[1] for x in test_preds]
            except:
                test_preds = model.predict(X_test)
            all_auc.append(roc_auc_score(y_test, test_preds))
        curr_score = np.array(all_auc).mean()

    except:
        stat = STATUS_FAIL
    print("\tAUC {0}\n\n".format(curr_score))
    return {'loss': 1 - curr_score, 'status': stat}


def optimize(trials):
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=20)
    print('best params:')
    print(best)


warnings.filterwarnings("ignore")

X_unlabled = load_data()

preg_tagged_data_reader = tagged_data_reader()
X = X_unlabled.merge(preg_tagged_data_reader.read_all(),on='drugBank_id',how='inner')
y = X['preg_class']
del X['drugBank_id']
del X['preg_class']
y = pd.DataFrame(np.array(transform_labels(y)))

trials = Trials()#Trials object where the history of search will be stored
normalize_type_dict = ['tree', 'forest']
sample_type_dict = ['uniform', 'weighted']


space = {
    'n_estimators': hp.uniformint('n_estimators', 50, 150),  # hp.normal
    #'num_k_best' : hp.qnormal('num_k_best', 500, 300,1),
     'eta': hp.quniform('eta', 0.01,0.3,0.01),
    # 'skip_drop': hp.quniform('skip_drop', 0.0, 1.0, 0.05),#only dart
    # 'rate_drop': hp.quniform('rate_drop', 0.0, 1.0, 0.05),#only dart
    # 'normalize_type': hp.choice('normalize_type', ['tree', 'forest']),  # only dart
    # 'sample_type': hp.choice('sample_type', ['uniform', 'weighted']),  # only dart
   # 'max_depth': hp.uniformint('max_depth', 1, 50),
#    'min_child_weight': hp.uniformint('min_child_weight', 50, 200),
   # 'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    #'gamma': hp.quniform('gamma', 0.1, 10, 0.01),
  #  'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    #'objective': hp.choice('objective', ['binary:logistic']),
    #'booster': hp.choice('booster', ['dart']),
 #   'nthread': 6,
  #  'silent': 1
}

if __name__ == "__main__":
    optimize(trials)