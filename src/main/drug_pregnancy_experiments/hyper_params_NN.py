import os
import warnings

import numpy
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, STATUS_FAIL
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
            model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(**params)
    )
            val = cross_val_score(model , X_cv, y, cv=kf, n_jobs=4, scoring='roc_auc')
            all_auc.extend(val)

        curr_score = np.array(all_auc).mean()

    except:
        stat = STATUS_FAIL

    print("\tAUC {0}\n\n".format(curr_score))
    return {'loss': 1 - curr_score, 'status': stat}


def optimize(trials):
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=300)
    print('best params:')
    print(best)


warnings.filterwarnings("ignore")

X_unlabled = load_data()

cols = list(X_unlabled.columns)
cols.remove('drugBank_id')
#
# sums = X_unlabled[cols].astype(bool).sum()
# limit = 3#len(raw_X)*0.0005
# sums = sums[sums>=limit]
# cols = list(sums.index)
# cols.append('drugBank_id')
# #raw_tagged_data = raw_tagged_data.loc[:,sums.index]
# X_unlabled = X_unlabled.loc[:,cols]
#

preg_tagged_data_reader = tagged_data_reader()
X_cv = X_unlabled.merge(preg_tagged_data_reader.read_all(),on='drugBank_id',how='inner')
y_cv = X_cv['preg_class']
del X_cv['drugBank_id']
del X_cv['preg_class']

y = numpy.ravel(pd.DataFrame(transform_labels(y_cv)).values)
trials = Trials()#Trials object where the history of search will be stored


space = {
    'n_neighbors': hp.uniformint('n_neighbors', 1, 200),  # hp.normal
      'weights': 'distance',#hp.choice('weights', ['distance']),
    'n_jobs': 4,
    'p': 1#hp.choice('p', [1,2,3]),
}

if __name__ == "__main__":
    optimize(trials)
