import operator
from copy import copy

import sklearn
from catboost import CatBoostClassifier
from keras import regularizers
from scipy.optimize import fmin
from scipy.sparse import csr_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier, \
    BaggingClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, RFE
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler, MinMaxScaler, FunctionTransformer, \
    PolynomialFeatures
from sklearn.svm import LinearSVC, SVC, NuSVC
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive
from sklearn.ensemble import StackingClassifier


class preg_scorer():
    def __init__(self,threshold=0.36):

        self.results = {'Name':[], 'AUC':[],'Kappa':[],'Threshold':[],'Itertion':[],'Modalities':[],'CV':[],'Precision':[], 'Recall':[],'F1-score':[],'Specificity':[],'Sensitivity':[] }
        self.threshold = threshold

    def as_df(self):
        c = dict(self.results)
        del c['Modalities']
        return pd.DataFrame(c)

    def get_best_mod(self,column_to_max='AUC'):
        base_mod = list(self.as_df().groupby('Name').mean().sort_values(by=column_to_max, ).index[-1].split('*')[0].split(';'))
        new_auc = self.as_df().groupby('Name').mean().sort_values(by=column_to_max,)[column_to_max][-1]
        return base_mod,new_auc

    def add_score(self,modalities,clfName,y_test,test_preds,rep,cv_i,best_threshold):
        auc = roc_auc_score(y_test, test_preds)
        self.results['Modalities'].append(modalities)
        self.results['Name'].append(';'.join(modalities) +'*'+clfName)
        self.results['CV'].append(cv_i)
        self.results['Itertion'].append(rep)
        self.results['Threshold'].append(best_threshold)
        self.results['AUC'].append(auc)
        test_preds_class = [0 if x < self.threshold else 1 for x in test_preds]

        report = classification_report(y_test, test_preds_class, output_dict=True)

        self.results['Precision'].append(report['0']['precision'])
        self.results['Recall'].append(report['0']['recall'])  # ==sensitivity
        self.results['F1-score'].append(report['0']['f1-score'])

        tn, fp, fn, tp = confusion_matrix(y_test, test_preds_class).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        t = 0.35
        kappa = sklearn.metrics.cohen_kappa_score(y_test, [0 if x < t else 1 for x in test_preds])
        self.results['Kappa'].append(kappa)
        self.results['Specificity'].append(specificity)
        self.results['Sensitivity'].append(sensitivity)

    def find_best_threshold(self,y_test,test_preds):
        best_thr = None
        best_val = 0
        for x0 in range(10):
            currMinres = fmin(thr_to_accuracy, args=(y_test, test_preds), x0=x0 / 10, disp=False,
                              full_output=True)
            if best_val < -currMinres[1]:
                best_val = -currMinres[1]
                best_thr = currMinres[0]
        return best_thr


def thr_to_accuracy(thr, Y_test, predictions):
    return -f1_score(Y_test, np.array(predictions > thr, dtype=np.int))

labels_array=['Safe','Limited']
def transform_labels(data):
    return [labels_array.index(x) for x in data]

def getlr():
    lr2 = LogisticRegression(C=20.0, dual=False, penalty='l2')  # .866783
    return lr2

def getnb():
    nb = BernoulliNB(alpha=0.01, fit_prior=False)  # .864
    return nb

def getRfeNb():
    nb2 = make_pipeline(
        RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.3, n_estimators=100),
            step=0.8500000000000001),
        BernoulliNB(alpha=0.01, fit_prior=False)
    )  # 0.873 but slow
    return nb2

def getKNN():
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(**{'n_jobs': 4, 'n_neighbors': 20, 'p': 1, 'weights': 'distance'})
    )
    return knn
def getAllClassifiers():

    return [('Extra trees optimized params',getExtraTreesOptimized()),
            #('Extra trees', getExtraTreesClassifierRaw())
            ]
    # return [
    #     ('Logistic regression', getlr()),
    #     ('Feature selection + Naive Bayes',getRfeNb()),
    #     #('ExtraTreesClassifier', getExtraTreesClassifierRaw()),
    #     ('ExtraTreesClassifier', getExtraTreesOptimized()),
    #     ('Normalization + KNN', getKNN()),
    #     ('Voting of all',getVoting())
    # ]

def getVoting():
    best = VotingClassifier(estimators=[
        ('Logistic regression', getlr()),
        ('Feature selection + Naive Bayes',getRfeNb()),
        ('ExtraTreesClassifier',getExtraTreesOptimized()),
        ('Normalization + KNN', getKNN()),
    ],voting='soft')
    return best

def getExtraTreesClassifierRaw():
    return ExtraTreesClassifier()

def getExtraTreesOptimized():
    #params = {'max_features': 80, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 547, 'n_jobs': 4} #OLD, 10 repeats: 0.869
    #params = {'max_features': 58, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 585}
    #params = {'max_features': 50, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 654, 'n_jobs': 4} # AUC 0.8795179924242423
    #{'max_features': 79, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 519, 'n_jobs': 4} #AUC 0.879369128787879
    #{'max_features': 50, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 673, 'n_jobs': 4} # AUC 0.8791486742424242
    #params ={'max_features': 62, 'min_samples_leaf': 10, 'min_samples_split': 4, 'n_estimators': 727, 'n_jobs': 4}

    #params = {'max_features': 35, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 550, 'n_jobs': 4} #0.88~
    #params = {'max_features': 39, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 806,'n_jobs': 4}  # 0.8881
    #params = {'max_features': 33, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 836, 'n_jobs': 4}
    params = {'max_features': 31, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 874, 'n_jobs': 4}
    #params ={'max_features': 32, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 387}
    return ExtraTreesClassifier(**params)



def get_keras_model():
    from keras.layers import Input, Dense, Dropout
    from keras.models import Model

    # This returns a tensor
    inputs = Input(shape=(3505,))
    dr_rate=0.05
    reg_rate=0.0001
    size=1000
    # a layer instance is callable on a tensor, and returns a tensor
    dr = Dropout(dr_rate)(inputs)
    final_dr =dr
    for i in range(10):
        dense = Dense(size, activation='relu',kernel_regularizer=regularizers.l1_l2(reg_rate))(dr)
        final_dr = Dropout(dr_rate)(dense)
    predictions = Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l1_l2(reg_rate))(final_dr)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def run_cross_DB_experiment(X_train,X_test,y_train,y_test):
    results_collector = preg_scorer()

    for clfName,clf in getAllClassifiers():
        #clf = getExtraTrees()  # xgb.XGBClassifier(  )
        clf.fit(X_train.astype(int), y_train)  # safe=0, 1=limited
        try:
            test_preds = clf.predict_proba(X_test.values)
            test_preds = [x[1] for x in test_preds]
        except:
            test_preds = clf.predict(X_test.values)
        best_thr = results_collector.find_best_threshold(y_test, test_preds)
        results_collector.add_score(clfName, y_test, test_preds, 0, 0, best_thr[0])
    return  results_collector.as_df()

def run_CV_experement(raw_tagged_data, y, repeats=10):
    results_collector = preg_scorer()
    #results = {'AUC':[], 'Name':[], 'Threshold':[],'Itertion':[],'CV':[],'Precision':[], 'Recall':[],'F1-score':[],'Specificity':[],'Sensitivity':[] }
    for rep in range(repeats):
        kf = StratifiedKFold(n_splits=10, random_state=rep, shuffle=True)
        for cv_i, (train_index, test_index) in enumerate(kf.split(raw_tagged_data, y)):
            for clfName,clf in getAllClassifiers():
                X_train, X_test = raw_tagged_data.iloc[train_index], raw_tagged_data.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                X_test = X_test.fillna(X_train.mean())
                X_train = X_train.fillna(X_train.mean())

                clf.fit(X_train.values, y_train)
                try:
                    test_preds= clf.predict_proba(X_test.values)
                    test_preds = [x[1] for x in test_preds]
                except:
                    test_preds = clf.predict(X_test.values)
                best_thr = results_collector.find_best_threshold(y_test,test_preds)
                results_collector.add_score( clfName, y_test, test_preds, rep, cv_i, best_thr[0])

        print('done rep',rep)

    return results_collector.as_df()
