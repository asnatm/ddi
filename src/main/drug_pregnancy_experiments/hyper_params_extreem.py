import os
import warnings

import numpy
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

    try:
        all_auc = []
        for i in range(10):
            kf = StratifiedKFold(n_splits=10, random_state=i, shuffle=True)
            val = cross_val_score(ExtraTreesClassifier(**params), X_cv, y, cv=kf, n_jobs=4, scoring='roc_auc')
            all_auc.extend(val)
        curr_score = np.array(all_auc).mean()

    except:
        stat = STATUS_FAIL

    print("\tAUC {0}\n\n".format(curr_score))
    return {'loss': 1 - curr_score, 'status': stat}


def optimize(trials):
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=1000)
    print('best params:')
    print(best)


warnings.filterwarnings("ignore")


X_unlabled = load_data()
preg_tagged_data_reader = tagged_data_reader()
X_cv = X_unlabled.merge(preg_tagged_data_reader.read_all(),on='drugBank_id',how='inner')
y_cv = X_cv['preg_class']
del X_cv['drugBank_id']
del X_cv['preg_class']

y = numpy.ravel(pd.DataFrame(transform_labels(y_cv)).values)

trials = Trials()#Trials object where the history of search will be stored
normalize_type_dict = ['tree', 'forest']
sample_type_dict = ['uniform', 'weighted']


space = {
    'n_estimators': hp.uniformint('n_estimators', 50, 1000),  # hp.normal
     #'ccp_alpha': hp.quniform('ccp_alpha', 0.0, 0.005, 0.001),
      #'bootstrap': hp.choice('bootstrap', [True, False]),
     #'max_depth': hp.uniformint('max_depth', 1, 200),
     'min_samples_split': hp.uniformint('min_samples_split', 2, 10),
      'min_samples_leaf': hp.uniformint('min_samples_leaf', 1, 10),
     'max_features': hp.uniformint('max_features', 30, 60),
    'n_jobs': 4,
}

if __name__ == "__main__":
    optimize(trials)
    #
    # n_estimators: Any = 100,
    # criterion: Any = "gini",
    # max_depth: Any = None,
    # min_samples_split: Any = 2,
    # min_samples_leaf: Any = 1,
    # min_weight_fraction_leaf: Any = 0.,
    # max_features: Any = "auto",
    # max_leaf_nodes: Any = None,
    # min_impurity_decrease: Any = 0.,
    # min_impurity_split: Any = None,
    # bootstrap: Any = False,
    # oob_score: Any = False,
    # n_jobs: Any = None,
    # random_state: Any = None,
    # verbose: Any = 0,
    # warm_start: Any = False,
    # class_weight: Any = None,
    # ccp_alpha: Any = 0.0,
    # max_samples: Any = None) -> None