import itertools
import os
import pickle
import random
import tensorflow as tf
import keras.backend as K
import networkx as nx
import numpy as np
from tensorflow import keras
from keras import Input, Model
from keras.layers import Embedding, Add, Dropout, Multiply, Concatenate, Flatten, Dense, Dot
from keras.optimizers import Adam
from keras.regularizers import l2
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics import roc_auc_score
from src.data_readers.d2d_releases_reader import d2d_releases_reader
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import csv
import json


from src.data_readers.dataset_dates import d2d_versions_metadata
from src.drug_classification.single_drug_features_creator import DrugBank_drug_feature_creator
from src.main.data_migration.table_names import get_DB_connection, AMFP_schema, combine_table_name_version, AMFP_table, \
    drugBank_id

os.chdir('..\\..\\..')

USE_DB = False
SAVE_DRUG_NAMES = False
MODEL_TRAIN = True
ADD_TEXT = True
YOUR_JSON_PATH='F:\\Drugs\\Concept_embedding2020\\concept_cbow.json'
YOUR_BIOCONCEPTVEC_PATH = "F:\\Drugs\\Concept_embedding2020\\bioconceptvec_word2vec_cbow.bin"
#DRUG_CONCEPT_NAMES = "F:\\Drugs\\Code\\DDI_prediction_mm\\multimodal_learning-master\\output\\data\\drug_names5.1.5_con.csv"   #word embedding
DRUG_CONCEPT_NAMES = "F:\\Drugs\\Code\\DDI_prediction_mm\\multimodal_learning-master\\output\\data\\drug_names5.1.5_convec_2021tot_tst.csv"   #concept embedding
ONLY_TEXT_VALID = True  #use only drugs with valid embedding
RANDOM_BASELINE = False
RANDOM_POSITIVE_RATIO = 0.144
ONLY_TEXT = False
MIN_COUNT = 10
RARE_MAX = 3
CONCEPT_NAME_F = 'concept_mesh'  #'concept_mesh'  #other option 'concept_name'
USE_BERT = True
BERT_MODEL_PATH = "F:\\Drugs\\Code\\preg\\multimodal_learning-pregnancy_safety_prediction\\multimodal_learning-pregnancy_safety_prediction\\pickles\\data\\input\\drug_names5.1.5_bert_emb_3.csv"
if USE_BERT:
    CONCEPT_NAME_F = 'name'
    DRUG_CONCEPT_NAMES = BERT_MODEL_PATH
#NO_TEXT = True #we build two models one only text one only interaction

def array_to_dict(array):
    return dict([(v, i) for i, v in enumerate(array)])

def isNaN(string):
        return string != string


class drug_evaluator():
    def __init__(self, drugs_array, interactions_newer, interactions_older = None):
        self.drugs_array = drugs_array
        self.interactions_newer = interactions_newer
        self.interactions_older = interactions_older
        self.test_positives = 0
        self.test_positives_ratio = 0.0

    @staticmethod
    def get_train_validation_sets(m):
        train_and_val_tuples = drug_evaluator.get_nnz_tuples_from_marix(m, zeros=False)
        validation_tuples,train_tuples = [],[]
        print('total train+validatoin tuples:', len(train_and_val_tuples))
        train_ratio = 0.9
        for t in train_and_val_tuples:
            if random.uniform(0, 1) < train_ratio:
                train_tuples.append(t)
            else:
                validation_tuples.append(t)
        print('train tuples:', len(train_tuples))
        print('validation tuples:', len(validation_tuples))
        return train_tuples,validation_tuples

    @staticmethod
    def get_nnz_tuples_from_marix(m,zeros):

        """returns <x,y> collection of the test tuples. The test are cells containing zero in the original matrix"""
        if zeros:
            res = [i for (i, v) in np.ndenumerate(m) if
                   v == 0 and i[0] > i[1]] #get the cells of matrix in ascending order of cell value
        else:
            res = [i for (i, v) in np.ndenumerate(m) if
                    i[0] > i[1]]  # get the cells of matrix in ascending order of cell value
        return res

    @staticmethod
    def get_nnz_tuples_from_marix_rare(m, zeros,rare_drugs):

        """returns <x,y> collection of the test tuples. The test are cells containing zero in the original matrix"""
        if zeros:
            res = [i for (i, v) in np.ndenumerate(m) if
                 v == 0 and i[0] in rare_drugs] #get the cells of matrix in ascending order of cell value
        else:
            res = [i for (i, v) in np.ndenumerate(m) if
                  i[0] in rare_drugs]  # get the cells of matrix in ascending order of cell value
        return res

    def print_data_split_summary(self,m_full, train_tuples, validation_tuples, test_tuples):
        count_i_train = 0
        count_i_validation = 0
        count_i_test = 0
        for t in train_tuples:
            if m_full[t[0], t[1]] > 0:
                count_i_train += 1
        for t in validation_tuples:
            if m_full[t[0], t[1]] > 0:
                count_i_validation += 1
        for t in test_tuples:
            if m_full[t[0], t[1]] > 0:
                count_i_test += 1
        print('train total: %d, train interactions: %d, train ratio: %f' % (
            len(train_tuples), count_i_train, count_i_train / len(train_tuples) if len(train_tuples)>0 else 0))
        print('validation total: %d, validation interactions: %d, validation ratio: %f' % (
            len(validation_tuples), count_i_validation, count_i_validation / len(validation_tuples) if len(validation_tuples)>0 else 0))
        print('test total: %d, test interactions: %d, test ratio: %f' % (
            len(test_tuples), count_i_test, count_i_test / len(test_tuples) if len(test_tuples)>0 else 0))

    def create_data_split_tuples(self, train_ratio = 0.7, validation_ratio = 0.15,test_ratio = 0.15):
        """generate train,test and validation sets using the number of drugs
        The sets are represented as tuples of the indexes in the matrix <x,y> where x<y"""
        n = len(self.drugs_array)
        print('number of drugs: %d' % n)
        assert train_ratio + validation_ratio + test_ratio == 1

        all_tuples = [(x, y) for x in range(n) for y in range(n) if x < y] #just upper half of the matrix
        train_tuples,validation_tuples ,test_tuples = [],[],[]
        print('number of cells in upper trainagle: %d' % len(all_tuples))

        for t in all_tuples:
            r = random.uniform(0, 1)
            if r < train_ratio:
                train_tuples.append(t)
            elif r < train_ratio + validation_ratio:
                validation_tuples.append(t)
            else:
                test_tuples.append(t)

        print('Only upper half: Train size: %d, validation size: %d, test size %d' % (len(train_tuples), len(validation_tuples), len(test_tuples)))

        # coding the drugs to codes now. this is not neede at the moment, but if we would like to put some
        #more logic into the splitting it will be needed
        assert len(train_tuples)==len(set(train_tuples)) and len(test_tuples)==len(set(test_tuples)) and len(validation_tuples)==len(set(validation_tuples))
        return train_tuples,validation_tuples,test_tuples, self.drugs_array

    def create_train_test_ratio_split(self, train_ratio=0.85,validation_ratio = 0, test_ratio=0.15):
        """
        returns a split of the data using dictionaries.
        The dics are symmetric: y in ans[x] -> x in ans[y]
        a mapping of index to drug name is also given as an array
        """
        assert train_ratio+validation_ratio+test_ratio==0
        train_drug_to_interactions ,test_drug_to_interactions,validation_drug_to_interactions   = {},{},{}
        train_tuples, validation_tuples, test_tuples, i2d = self.create_data_split_tuples(train_ratio,validation_ratio,test_ratio)
        d2i = array_to_dict(i2d)
        train_tuples, validation_tuples, test_tuples = set(train_tuples),set(validation_tuples),set(test_tuples)

        for drug1,drug1_interactions in self.interactions_newer.items():
            for drug2 in drug1_interactions:
                assert drug1 != drug2 #just making sure again
                drug1_index = d2i[drug1]
                drug2_index = d2i[drug2]
                t = (drug1_index,drug2_index)
                if t in train_tuples or tuple(reversed(t)) in train_tuples:
                    train_drug_to_interactions.setdefault(drug1,[]).append(drug2)
                    assert drug2 not in train_drug_to_interactions or drug1 not in train_drug_to_interactions[drug2], 'the drug is already in the list. doing to insert it again'
                    train_drug_to_interactions.setdefault(drug2, []).append(drug1)
                elif t in validation_tuples or tuple(reversed(t)) in validation_tuples:
                    validation_drug_to_interactions.setdefault(drug1,[]).append(drug2)
                    assert drug2 not in validation_drug_to_interactions or drug1 not in validation_drug_to_interactions[drug2], 'the drug is already in the list. doing to insert it again'
                    validation_drug_to_interactions.setdefault(drug2, []).append(drug1)
                else:
                    assert t in test_tuples, 'drug wasnt put anywhere'
                    test_drug_to_interactions.setdefault(drug1,[]).append(drug2)
                    assert drug2 not in test_drug_to_interactions or drug1 not in test_drug_to_interactions[drug2], 'the drug is already in the list. doing to insert it again'
                    test_drug_to_interactions.setdefault(drug2, []).append(drug1)
        return train_drug_to_interactions, validation_drug_to_interactions, test_drug_to_interactions, i2d

    def print_roc(self,overall_fpr_tpr,ax,style=None):
        #fig = plt.figure(figsize=(5.2, 3.9), dpi=600)
        #ax = fig.add_subplot(111)
        for j, ftp_tpr in enumerate(overall_fpr_tpr):
            #ax.plot(ftp_tpr[0], ftp_tpr[1], label=ftp_tpr[2], linestyle=style[j][0],color = style[j][1],linewidth=style[j][2])
            ax.plot(ftp_tpr[0], ftp_tpr[1], label=ftp_tpr[2])
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC")
        ax.legend(loc="upper left", bbox_to_anchor=(1,1))


    def print_evaluation(self, preds,ax,style=None,max_k=100,title="",legend_loc=4):
        #fig = plt.figure(figsize=(5.2, 7.8), dpi=600)
        #ax = fig.add_subplot(111)
        for i,values in enumerate(preds):
            data=values[0][:max_k]
            #print(f"name:{name}, precision@k: {data}")
            #ax.plot(range(1,len(data)+1),data,label=values[1],linestyle=style[i][0],color = style[i][1],linewidth=style[i][2])
            ax.plot(range(1, len(data) + 1), data, label=values[1])
        ax.set_xlabel("n")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        #ax.legend( loc=legend_loc)


    def get_precision_per_drug(self, m_test, predictions,k=5):
        n_drugs = m_test.shape[0]
        per_drug_pred = [[None for i in range(k)] for j in range(n_drugs)]
        cnt_interactions_per_drug = [0 for i in range(n_drugs)]
        for i,t in enumerate(predictions):
            drug_ids = t[0]
            d1 = drug_ids[0]
            d2 = drug_ids[1]
            assert d1!=d2
            if cnt_interactions_per_drug[d1]<k:
                per_drug_pred[d1][cnt_interactions_per_drug[d1]] = d2
                cnt_interactions_per_drug[d1]+=1
            if cnt_interactions_per_drug[d2] < k:
                per_drug_pred[d2][cnt_interactions_per_drug[d2]] = d1
                cnt_interactions_per_drug[d2] += 1

        per_drug_precision= [[None for i in range(k)] for j in range(n_drugs)]
        per_drug_recall = [[None for i in range(k)] for j in range(n_drugs)]
        for d in range(n_drugs):
            tp = 0
            s = np.sum(m_test[d])
            for i in range(1,k+1):
                if per_drug_pred[d][i-1] is not None:
                    if m_test[d,per_drug_pred[d][i-1]] !=0:
                        tp+=1
                    per_drug_precision[d][i-1]=tp/i
                    if s > 0:
                        per_drug_recall[d][i-1] = tp/s
                    else:
                        per_drug_recall[d][i-1] = 1.0
                else:
                    assert False, "drugs with no prediction"
        res = np.average(per_drug_precision,axis=0)
        res_recall = np.average(per_drug_recall,axis=0)
        #print(res)
        return res,res_recall
        #print(cnt_interactions_per_drug)

    def get_precision_per_drug_rare(self, m_test, predictions,test_rare,rare_drugs,k=5):
        n_drugs = m_test.shape[0]
        test_rare_set = set(test_rare)
        n_drugs_rare = len(rare_drugs)
        rare_drugs_dict = {}
        for ind,dr in enumerate(rare_drugs):
            rare_drugs_dict[dr] = ind

        #rare_drugs_set = set(rare_drugs)
        per_drug_pred = [[None for i in range(k)] for j in range(n_drugs_rare)]
        cnt_interactions_per_drug = [0 for i in range(n_drugs_rare)]
        for i,t in enumerate(predictions):
            drug_ids = t[0]
            d1 = drug_ids[0]
            d2 = drug_ids[1]
            assert d1!=d2
            if (drug_ids[0] ,drug_ids[1]) not in test_rare_set:
                continue
            if d1 in rare_drugs:
                if cnt_interactions_per_drug[rare_drugs_dict[d1]]<k:
                    per_drug_pred[rare_drugs_dict[d1]][cnt_interactions_per_drug[rare_drugs_dict[d1]]] = d2
                    cnt_interactions_per_drug[rare_drugs_dict[d1]]+=1
            if d2 in rare_drugs:
                if cnt_interactions_per_drug[rare_drugs_dict[d2]] < k:
                    per_drug_pred[rare_drugs_dict[d2]][cnt_interactions_per_drug[rare_drugs_dict[d2]]] = d1
                    cnt_interactions_per_drug[rare_drugs_dict[d2]] += 1

        per_drug_precision= [[None for i in range(k)] for j in range(n_drugs_rare)]
        per_drug_recall = [[None for i in range(k)] for j in range(n_drugs_rare)]
        for d in range(n_drugs):
            tp = 0
            if d not in rare_drugs:
                continue
            di = rare_drugs_dict[d]
            s = np.sum(m_test[d])
            for i in range(1,k+1):
                if per_drug_pred[di][i-1] is not None:
                    if (m_test[d,per_drug_pred[di][i-1]] !=0) or (m_test[per_drug_pred[di][i-1],d] !=0):
                        tp+=1
                    per_drug_precision[di][i-1]=tp/i
                    if s > 0:
                        per_drug_recall[di][i - 1] = tp / s
                    else:
                        print("problem rare drug with not test predictions")
                else:
                    assert False, "drugs with no prediction xx"
        print('remove precision_per_durg len per_drug_precision '+str(len(per_drug_precision))+', rare drugs ',str(n_drugs_rare))
        res = np.average(per_drug_precision,axis=0)
        res_recall = np.average(per_drug_recall,axis=0)
        #print(res)
        return res,res_recall
        #print(cnt_interactions_per_drug)



    def get_precision_recall_rare(self, m_test, predictions, test_tuples,rare_drugs):
        print('remove pre_recall rare ')
        N,P=0,0
        test_tuples_set = set(test_tuples)
        for t in test_tuples_set:
            #print(t)
            # if (t[0] < t[1]):
            #     continue
            if m_test[t[0], t[1]] > 0:
                P+=1
            else:
                N+=1
        print('Rare Positives: %d, Negatives: %d' % (P,N))
        print('Rare Positive ratio: %f' % (P/(N+P)))
        self.test_positives = P
        self.test_positive_ratio = P/(N+P)

        print('before loop')
        precision_at_k, recall_at_k, class_correct  = [],[], []
        t=0
        test_tuples_set = set(test_tuples)
        ii = 0
        for i,tuple in enumerate(predictions):
            drug_ids = tuple[0]
            if (drug_ids[0] ,drug_ids[1]) not in test_tuples_set:
                #print('remove skip drugs not rare')
                continue
                #assert False, 'edges were predicted which are from the training set' + str(drug_ids[0]) +" ," + str(drug_ids[1])
            # if (drug_ids[0]) not in rare_drugs:
            ##     continue
            #test_tuples_set.remove((drug_ids[0], drug_ids[1]))
            if m_test[drug_ids[0], drug_ids[1]] > 0:
                t+=1
                class_correct.append(True)
            else:
                class_correct.append(False)
            precision_at_k.append(t/(ii+1))
            recall_at_k.append(t/P)
            ii = ii + 1
        print('rare precision at 100:',str(precision_at_k[100]))
        #assert len(test_tuples_set)==0, f'unpredicted interactions, {test_tuples_set}'
        print('rare precision @ cutoff: %f, recall @ cutoff: %f, cutoff: %d' % (precision_at_k[P-1],recall_at_k[P-1],P))
        return precision_at_k,recall_at_k, class_correct
        # predictions = predictions[:P] #the amount we predict is the amount of Trues in the test set TODO: change this
        #
        # TP = []
        # FP = []
        # for drug_ids in predictions:
        #     if m_test[drug_ids[0], drug_ids[1]]>0:
        #         #assert (drug_ids[0],drug_ids[1]) in test_tuples #it is ok, but it takes a very long time
        #         assert m_test[drug_ids[0], drug_ids[1]] != 0
        #         TP.append((drug_ids[0],drug_ids[1]))
        #     else:
        #         assert m_test[drug_ids[1], drug_ids[0]] == 0
        #         FP.append((drug_ids[0],drug_ids[1]))
        # print('tp', len(TP), 'fp', len(FP), 'precision:', len(TP) / (len(TP) + len(FP)))
        # print('count in test',len(test_tuples))
        # assert P == len(TP)+len(FP)
        # #print('TP sample:', TP[:100])
        # #print('FP smaple:', FP[:100])
        # return TP,FP


    def get_precision_recall(self, m_test, predictions, test_tuples):
        N,P=0,0
        for t in test_tuples:
            if m_test[t[0], t[1]] > 0:
                P+=1
            else:
                N+=1
        print('Positives: %d, Negatives: %d' % (P,N))
        print('Positive ratio: %f' % (P/(N+P)))
        self.test_positives = P
        self.test_positive_ratio = P/(N+P)


        precision_at_k, recall_at_k, class_correct  = [],[], []
        t=0
        test_tuples_set = set(test_tuples)
        for i,tuple in enumerate(predictions):
            drug_ids = tuple[0]
            if (drug_ids[0] ,drug_ids[1]) not in test_tuples_set:
                assert False, 'edges were predicted which are from the training set' + str(drug_ids[0]) +" ," + str(drug_ids[1])
            test_tuples_set.remove((drug_ids[0], drug_ids[1]))
            if m_test[drug_ids[0], drug_ids[1]] > 0:
                t+=1
                class_correct.append(True)
            else:
                class_correct.append(False)
            precision_at_k.append(t/(i+1))
            recall_at_k.append(t/P)
        print('precision at 100:',str(precision_at_k[100]))
        assert len(test_tuples_set)==0, f'unpredicted interactions, {test_tuples_set}'
        print('precision @ cutoff: %f, recall @ cutoff: %f, cutoff: %d' % (precision_at_k[P-1],recall_at_k[P-1],P))
        return precision_at_k,recall_at_k, class_correct
        # predictions = predictions[:P] #the amount we predict is the amount of Trues in the test set TODO: change this
        #
        # TP = []
        # FP = []
        # for drug_ids in predictions:
        #     if m_test[drug_ids[0], drug_ids[1]]>0:
        #         #assert (drug_ids[0],drug_ids[1]) in test_tuples #it is ok, but it takes a very long time
        #         assert m_test[drug_ids[0], drug_ids[1]] != 0
        #         TP.append((drug_ids[0],drug_ids[1]))
        #     else:
        #         assert m_test[drug_ids[1], drug_ids[0]] == 0
        #         FP.append((drug_ids[0],drug_ids[1]))
        # print('tp', len(TP), 'fp', len(FP), 'precision:', len(TP) / (len(TP) + len(FP)))
        # print('count in test',len(test_tuples))
        # assert P == len(TP)+len(FP)
        # #print('TP sample:', TP[:100])
        # #print('FP smaple:', FP[:100])
        # return TP,FP


    def create_train_matrix(self,m_full,train_tuples,validation_tuples,test_tuples):
        m_train = lil_matrix(m_full)
        # for t in validation_tuples:
        #     m_train[t[0],t[1]] = 0
        #     m_train[t[1], t[0]] = 0
        for t in test_tuples:
            m_train[t[0],t[1]] = 0
            m_train[t[1], t[0]] = 0
        print('train matrix non zeros: %d' % m_train.nnz)
        return m_train.todense()




def create_train_validation_split(m_train,train_ratio = 0.9):
    train_and_val_tuples = drug_evaluator.get_nnz_tuples_from_marix(m_train, zeros=False)
    train_tuples, validation_tuples = [], []
    print('total train+validatoin tuples:', len(train_and_val_tuples))
    m = m_train.copy()
    for t in train_and_val_tuples:
        train_tuples.append(t)
        if random.uniform(0, 1) > train_ratio:
            validation_tuples.append(t)
            m[t[0], t[1]] = 0
            m[t[1], t[0]] = 0
    return train_tuples,validation_tuples, m




class drugs_nn_predictor():
    def __init__(self, m, test_tuples, validation_tuples=None, validation_target=None,name='AMF',propagation_factor=None, mul_emb_size = 128, dropout=0.3, epochs=5, batch_size=256, learning_rate=0.01,neg_per_pos_sample=1.0):
        #sub models: [similarity, GMF, MLP]
        # import tensorflow as tf
        # tf.set_random_seed(1)
        #super().__init__()
        self.predictions = None
        self.m = m.copy()
        self.name=name
        self.predictions_pickle_file_name=None
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.nn_predicted_mat=None
        self.model_file_name =None

        self.dropout = dropout
        self.epochs=epochs
        self.mul_emb_size = mul_emb_size
        self.test_tuples = test_tuples
        self.color = 'black'
        self.linestyle = '-'
        self.linewidth = 1
        self.validation_features=validation_tuples
        self.validation_target_att=validation_target
        m_train_array = np.squeeze(np.asarray(self.m))
        self.train_vector_list = dict([(x, m_train_array[x, :]) for x in range(self.m.shape[0])])
        self.propgation_factor=propagation_factor
        #features_creator = graph_features_creator(m)
        #self.features_dict = features_creator.get_normalized_feature_dict()
        #self.num_features = len(self.features_dict[(0, 0)])
        self.num_of_drugs = self.m.shape[0]
        self.neg_per_pos_sample=neg_per_pos_sample

        #features_file_path = r'pickles\filename.pickle'#TODO: create this file using code
        #with open(features_file_path, 'rb') as handle:
        #self.features_dict = pickle.load(handle)
        #self.num_feautres = len(next(iter(self.features_dict.items()))[1])

    def save_model(self):
        print("saving model ",self.model_file_name)
        try:
            self.model_emb.save(self.model_file_name)
        except:
            print("error saving model ",self.model_file_name)

    def load_model(self):
        try:
            print("cwd ",os.getcwd())
            self.model_emb =  keras.models.load_model(self.model_file_name)
        except:
            print("error loading model ", self.model_file_name)



    def get_sample_train_validation(self, train_pos, train_neg, validation_pos, validation_neg, neg_to_pos_ratio=1.0):
        if neg_to_pos_ratio is None:
            train, validation = train_pos + train_neg, validation_pos + validation_neg
        else:
            train = list()
            # validation = list()
            # random.shuffle(validation_pos);random.shuffle(validation_neg)
            # random.shuffle(train_pos);random.shuffle(train_neg)
            #         train = sample_each_drug_once(train_pos,train)
            #         train = sample_each_drug_once(train_neg,train)
            #         validation = sample_each_drug_once(validation_pos,validation)
            #         validation = sample_each_drug_once(validation_neg,validation)
            train = list(train_pos)
            # validation = list(validation_pos)
            if len(train_pos) * neg_to_pos_ratio < len(train_neg):
                train += random.sample(train_neg, int(len(train_pos) * neg_to_pos_ratio))
            else:
                print('not sampling due to increased number of positive samples')
                train += train_neg
                # validation += random.sample(validation_neg, len(validation_pos))
            validation = validation_pos + validation_neg

        train = [(x[0], x[1]) if random.random() > 0.5 else (x[1], x[0]) for x in
                 train]  # this is redundent now as shared layer is used
        validation = [(x[0], x[1]) if random.random() > 0.5 else (x[1], x[0]) for x in validation]
        return train, validation

    def create_pos_neg_instances(self,train_tuples, validation_tuples, m_train,m_validation):
        train_pos = [x for x in train_tuples if m_train[x[0], x[1]] == 1]
        train_neg = [x for x in train_tuples if m_train[x[0], x[1]] == 0]
        validation_pos = [x for x in validation_tuples if m_validation [x[0], x[1]] == 1]
        validation_neg = [x for x in validation_tuples if m_validation [x[0], x[1]] == 0]
        print(
            f'train pos: {len(train_pos)}, train neg: {len(train_neg)}, val pos: {len(validation_pos)}, val neg: {len(validation_neg)}')
        return train_pos, train_neg, validation_pos, validation_neg

    def fit(self):
        self.init_nn_model()
        self.fit_nn_model()
        print("remove no text")


    def get_embeddings(self):
        return [np.concatenate([a, b]) for a,b in zip(self.mult_dense.get_weights()[0],self.mlp.get_weights()[0])]

    def get_instances(self, tuples_sample, m):
        instance_features = []
        instance_features.append(np.array([t[0] for t in tuples_sample]))
        instance_features.append(np.array([t[1] for t in tuples_sample]))
        target_att = np.array([[m[t[0], t[1]]] for t in tuples_sample])
        return instance_features, target_att

    def init_nn_model(self):
        input_node_a = Input(shape=(1,), name='b')
        input_node_b = Input(shape=(1,), name='c')

        regularization = 0
        mlp_emb = Embedding(output_dim=1, name='MLP_embedding',input_dim=self.num_of_drugs,embeddings_regularizer=l2(regularization))
        self.mlp = mlp_emb
        emb_mlp1 = mlp_emb(input_node_a)
        emb_mlp2 = mlp_emb(input_node_b)
        l_mlp = Add()([emb_mlp1,emb_mlp2])

        mult_dense = Embedding(output_dim=self.mul_emb_size, name='GMF_embedding',embeddings_regularizer=l2(regularization),input_dim=self.num_of_drugs) #
        self.mult_dense = mult_dense
        emb_mult1 = mult_dense(input_node_a)
        emb_mult2 = mult_dense(input_node_b)
        dr_emb_mult1 = Dropout(self.dropout)(emb_mult1)
        dr_emb_mult2 = Dropout(self.dropout)(emb_mult2)
        mult = Dropout(0)(Multiply()([dr_emb_mult1, dr_emb_mult2]))
        final_layers = Concatenate(axis=-1)([x for i, x in enumerate([Flatten()(l_mlp), Flatten()(mult)])])
        main_output= Dense(1, activation='sigmoid')(final_layers)#the init is critical for the model to work
        model_emb = Model(inputs=[input_node_a,input_node_b], outputs=main_output)  # fixed_input
        model_emb.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['mae'])  # binary_crossentropy

        #from keras.utils.vis_utils import plot_model
        #import os
        #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        #plot_model(model_emb, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # self.model_gmf = Model(inputs=[input_node_a,input_node_b], outputs=mult_output) #fixed_input
        # self.model_gmf.compile(optimizer=Adam(),loss='binary_crossentropy', metrics=['accuracy','mae']) #binary_crossentropy
        #
        # self.model_mlp = Model(inputs=[input_node_a,input_node_b], outputs=n_con_output) #fixed_input
        # self.model_mlp.compile(optimizer=Adam(),loss='binary_crossentropy', metrics=['accuracy','mae']) #binary_crossentropy
        print(model_emb.summary())
        self.model_emb = model_emb

    def fit_nn_model(self):

        learning_rate = self.learning_rate
        batch_size = self.batch_size
        epochs=self.epochs
        train_tuples, validation_tuples,self.m_train = create_train_validation_split(self.m, train_ratio=0.8)  # if ratio = 1 then no validation
        #train_tuples, validation_tuples, self.m_train = create_train_validation_split_single_sample_per_drug(self.m,train_ratio=0.75)


        train_pos, train_neg, validation_pos, validation_neg = self.create_pos_neg_instances(train_tuples, validation_tuples,self.m_train, self.m)

        cnt_epoch=0
        current_learning_rate = learning_rate
        while epochs > cnt_epoch:
            cnt_epoch+=1
            print(f"Epoch number {cnt_epoch} with LR {current_learning_rate}")
            K.set_value(self.model_emb.optimizer.lr, learning_rate)
            # create sample instances#
            train_tuples_sample, validation_tuples_sample = self.get_sample_train_validation(train_pos, train_neg,
                                                                                             validation_pos, validation_neg,
                                                                                             neg_to_pos_ratio=self.neg_per_pos_sample)
            train_features, train_target_att = self.get_instances(train_tuples_sample,self.m_train)
            if self.validation_features==None:
                self.validation_features, self.validation_target_att = self.get_instances(validation_tuples_sample,self.m,)

            if self.validation_features!=None and len(self.validation_target_att)>0:
                self.model_emb.fit(x=train_features, y=train_target_att, batch_size=batch_size, epochs=1,verbose=2,validation_data=(self.validation_features, self.validation_target_att) ) # ,callbacks=[earlycurrent_learning_rateop]
                y_pred = self.model_emb.predict(self.validation_features,batch_size=50000)
                auc = roc_auc_score(self.validation_target_att, y_pred)
                print(f'auc: {auc}')
            else:
                self.model_emb.fit(x=train_features, y=train_target_att, batch_size=batch_size, epochs=1,
                                  verbose=2)  # ,validation_data=(self.validation_features, self.validation_target_att))  # ,callbacks=[earlycurrent_learning_rateop]

        self.w = self.mult_dense.get_weights()
        if sum(self.validation_target_att)>0:
            y_pred = self.model_emb.predict(self.validation_features)
            auc = roc_auc_score(self.validation_target_att, y_pred)
            best_auc = None
            best_x = None
            print(
                f"new evalm before: {auc}, {self.model_emb.evaluate(self.validation_features,self.validation_target_att,verbose=0)}")
            results = []
            for i in range(1,1+1):
                for x in [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
                    self.update_weights(orig_ratio=x,iterations=i)
                    y_pred = self.model_emb.predict(self.validation_features,batch_size=50000)
                    auc = roc_auc_score(self.validation_target_att, y_pred)
                    loss = self.model_emb.evaluate(self.validation_features, self.validation_target_att, verbose=0,batch_size=50000)[0]
                    print(f"new evalm {x}, {i}: {auc}, {loss}")
                    results.append((x,i,auc,loss))
                    if best_auc == None or auc>best_auc:
                        best_auc=auc
                        best_x=x
            print(f'best propagation AUC {best_auc}, best factpr: {best_x}')
            print("results=",results)
        if self.propgation_factor!=None:
            print(f'setting weights to {self.propgation_factor}')
            self.update_weights(orig_ratio=self.propgation_factor)
        print("DONE!")

    def update_weights(self,orig_ratio,iterations=1):
        self.mult_dense.set_weights(self.w)

        for x in range(iterations):
            #y_pred = self.model_emb.predict(self.validation_features, batch_size=100000)
            #first_loss = self.model_emb.evaluate(self.validation_features,self.validation_target_att, batch_size=100000,verbose=0)[0]
            #first_loss = roc_auc_score(self.validation_target_att, y_pred)
            w = self.mult_dense.get_weights()
            new_w =[]
            replaced_count = 0
            for v1 in range(self.num_of_drugs):

                new_node_emb = np.array(w[0][v1])
                new_w.append(new_node_emb)

            G = nx.from_numpy_matrix(self.m_train)
            for v1 in range(self.num_of_drugs):
                if len(G[v1])>0:
                    new_node_emb = np.array(w[0][v1])
                    other_nodes_w = np.zeros(len(new_node_emb)) #empty vector
                    total_weights=0
                    #print(other_nodes_w)
                    #w2 = 0
                    for v2 in G[v1]:
                        curr_weight = 1/len(G[v1])#
                        #w2+=len(G[v2])
                        total_weights+=curr_weight
                        other_nodes_w += curr_weight *w[0][v2]
                    #w1 = len(G[v1])
                    #w2 /= len(G[v1])
                    new_node_emb = new_node_emb*orig_ratio + (1-orig_ratio)*other_nodes_w/total_weights #   here the orig_ratio is 1-alpha from the paper.
                    #new_w.append(new_node_emb*orig_ratio + (1-orig_ratio)*other_nodes_w/total_weights)
                else:
                    new_node_emb = np.array(w[0][v1])
                    #new_w.append(np.array(w[0][v1]))
                #old_w = new_w[v1]
                #new_w[v1] = new_node_emb
                #self.mult_dense.set_weights(np.array([np.array(new_w)]))
                new_w[v1] = new_node_emb
            #     w[0][v1] = new_node_emb
            #     self.mult_dense.set_weights(w)
            #     #y_pred = self.model_emb.predict(self.validation_features,batch_size=100000)
            #     loss = self.model_emb.evaluate(self.validation_features,self.validation_target_att, batch_size=100000,verbose=0)[0]
            #     #auc = roc_auc_score(self.validation_target_att, y_pred)
            #     print(f'Loss for {v1}, {loss }')
            #
            #     w[0][v1] = new_w[v1] #new_w now have the original weight
            #     self.mult_dense.set_weights(w)
            #     if loss < first_loss:
            #         new_w[v1] = new_node_emb
            #         replaced_count+=1
            # print(f'replaced embedding: {replaced_count}')
            self.mult_dense.set_weights(np.array([np.array(new_w)]))

        # for x in range(iterations):
        #     w = self.mlp.get_weights()
        #     new_w =[]
        #     G = nx.from_numpy_matrix(self.m_train)
        #     for v1 in range(self.num_of_drugs):
        #         if len(G[v1])>0:
        #             new_node_emb = np.array(w[0][v1]) * orig_ratio
        #             for v2 in G[v1]:
        #                 new_node_emb += (1-orig_ratio)*(1/len(G[v1]))*w[0][v2]
        #             new_w.append(new_node_emb)
        #         else:
        #             new_w.append(np.array(w[0][v1]))
        #     self.mlp.set_weights(np.array([np.array(new_w)]))

            # for x in range(iterations):
            #     w = self.mult_bias.get_weights()
            #     new_w = []
            #     G = nx.from_numpy_matrix(self.m_train)
            #     for v1 in range(self.num_of_drugs):
            #         if len(G[v1]) > 0:
            #             new_node_emb = np.array(w[0][v1]) * orig_ratio
            #             for v2 in G[v1]:
            #                 new_node_emb += (1 - orig_ratio) * (1 / len(G[v1])) * w[0][v2]
            #             new_w.append(new_node_emb)
            #         else:
            #             new_w.append(np.array(w[0][v1]))
            #     self.mult_bias.set_weights(np.array([np.array(new_w)]))

            #print('bias values:',str(self.mult_bias.get_weights()))


    def predict(self):

        # import math
        # import train_generator
        # importlib.reload(train_generator)
        # from train_generator import d2d_generators

        # d2d_generators_object = d2d_generators(m_train,train_tuples,validation_tuples)

        batch_size = 100000
        # m_evaluation = m_train
        # evaluation_tuples = validation_tuples
        m_evaluation = self.m
        #evaluation_tuples = self.test_tuples  # test_tuples
        evaluation_tuples = [(x,y) for y in range(self.num_of_drugs) for x in range(y+1,self.num_of_drugs)]#self.test_tuples  # test_tuples
        print(f'evaluating {len(evaluation_tuples)} instances')
        eval_instances, _ = self.get_instances(evaluation_tuples,self.m)
        # preds_him = get_pred_him()
        print('done creating instances')
        preds = self.model_emb.predict(eval_instances,
                                  batch_size=batch_size)  # [preds_him[x[0],x[1]] for x in evaluation_tuples] #
        print('done predicting', len(preds))
        count = 0
        predictions = np.zeros((self.m.shape[0], self.m.shape[1]))
        for i in range(len(evaluation_tuples)):
            idx1 = evaluation_tuples[i][0]
            idx2 = evaluation_tuples[i][1]
            predictions[idx1, idx2] = preds[i][0]
            count += 1
        eval_instances, _ = self.get_instances([(x[1], x[0]) for x in evaluation_tuples],self.m)
        preds = self.model_emb.predict(eval_instances,
                                  batch_size=batch_size)  # [preds_him[x[1],x[0]] for x in evaluation_tuples]#
        for i in range(len(evaluation_tuples)):
            idx1 = evaluation_tuples[i][0]
            idx2 = evaluation_tuples[i][1]
            predictions[idx2, idx1] = preds[i][0]  # must be reversed...
            count += 1
        for i in range(self.m.shape[0]):
            for j in range(i + 1, self.m.shape[0]):
                from scipy import stats
                # new_score = stats.hmean([ max(0.000001,predictions[i,j]),max(0.000001,predictions[j,i]) ])
                new_score = (predictions[i, j] + predictions[j, i]) / 2
                # new_score = max(predictions[i,j],predictions[j,i])
                # new_score = predictions[j,i]
                predictions[i, j] = new_score
                predictions[j, i] = new_score
        for i in range(self.m.shape[0]):
            assert predictions[i, i] == 0



        # predictions2 = np.zeros((self.m.shape[0], self.m.shape[1]))
        # G = nx.from_numpy_matrix(self.m)
        # for v1,v2 in  [(i, j) for i in G.nodes() for j in G.nodes() if j>i]:
        #     nn_preds1 = []
        #     for z in G[v2]:
        #         nn_preds1.append(predictions[v1, z])
        #     nn_preds2 = []
        #     for z in G[v1]:
        #         nn_preds2.append(predictions[v2, z])
        #     predictions2[v1, v2] = 0.5*predictions[v1, v2] + 0.5*np.mean(np.nan_to_num([np.mean(nn_preds1),np.mean(nn_preds2)]))
        #     predictions2[v2, v1] = predictions2[v1, v2]

        #predictions = predictions2

        self.predicted_mat = predictions
        print('predicted: ', count)
        s = set([(x[0],x[1])for x in self.test_tuples])
        predictions = [(i, v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(predictions)],
                                                   reverse=True)]  # get the cells of matrix in ascending order of cell value
        print(1)
        predictions = [(t, v) for t, v in predictions
                       if (t[0], t[1]) in s ]  # just half of the matrix and predictions larger than 0
        if self.predictions_pickle_file_name!=None:
            nn_predicted_mat = self.predicted_mat
            pickling_on = open(self.predictions_pickle_file_name, "wb")
            pickle.dump(nn_predicted_mat, pickling_on)
            pickling_on.close()

        return predictions

class drugs_text_nn_predictor(drugs_nn_predictor):
#class drugs_text_nn_predictor():
    def __init__(self, m, test_tuples, validation_tuples=None, validation_target=None,name='AMF',propagation_factor=None,text_propagation_factor=None,text_propagation_factor2=None, mul_emb_size = 128, dropout=0.3, epochs=5, batch_size=256, learning_rate=0.01,neg_per_pos_sample=1.0):
        #sub models: [similarity, GMF, MLP]
        # import tensorflow as tf
        # tf.set_random_seed(1)
        #super().__init__()
        drugs_nn_predictor.__init__(self,m, test_tuples, validation_tuples=validation_tuples,
                           validation_target=validation_target, name='AMFP', propagation_factor=propagation_factor,mul_emb_size=mul_emb_size,dropout=dropout,epochs=epochs,batch_size=batch_size,learning_rate=learning_rate)
        #drugs_nn_predictor.__init__(self,m, test_tuples, validation_tuples=None, validation_target=None,name='AMF',propagation_factor=None, mul_emb_size = 128, dropout=0.3, epochs=5, batch_size=256, learning_rate=0.01,neg_per_pos_sample=1.0)
        if CONCEPT_NAME_F == 'concept_mesh':
            self.text_embedding_model = self.load_embedding_concept(YOUR_JSON_PATH)
            self.text_embedding_size = len(self.text_embedding_model['Chemical_MESH_C083544'])
        elif USE_BERT:
            self.text_embedding_model = self.load_embedding_bert(BERT_MODEL_PATH)
            self.text_embedding_size = self.text_embedding_model.loc['Lepirudin',:].shape[0]
        else:
            self.text_embedding_model = self.load_embedding(YOUR_BIOCONCEPTVEC_PATH,binary=True)
            self.text_embedding_size = self.text_embedding_model['lepirudin'].shape[0]

        self.drugs_concept_names = pd.read_csv(DRUG_CONCEPT_NAMES)

        self.drug_id_to_name = None
        self.i2d = None
        self.default_text_emb = self.set_default_text_embedding_vec()
        self.concept_embedding_matrix = None
        self.text_epochs = 20
        self.text_propgation_factor = text_propagation_factor
        self.text_propgation_factor2 = text_propagation_factor2
        #self.num_feautres = len(next(iter(self.features_dict.items()))[1])

    def set_drug_id_to_name(self,drug_id_to_name,i2d):
        self.drug_id_to_name = drug_id_to_name
        self.i2d = i2d

    def set_model_text(self,model_in):
        self.model_text = model_in

    def update_text_weights2(self,orig_ratio,iterations=1):
        '''
        update text of embedded model
        :param orig_ratio:
        :param iterations:
        :return:
        '''

        self.text_embedding_layer2.set_weights(self.tw2)
        max_drug = len(self.i2d)
        print(max_drug)
        for x in range(iterations):
            # y_pred = self.model_emb.predict(self.validation_features, batch_size=100000)
            # first_loss = self.model_emb.evaluate(self.validation_features,self.validation_target_att, batch_size=100000,verbose=0)[0]
            # first_loss = roc_auc_score(self.validation_target_att, y_pred)
            w = self.text_embedding_layer2.get_weights()
            new_w = []
            replaced_count = 0
            for v1 in range(max_drug):
                new_node_emb = np.array(w[0][v1])
                new_w.append(new_node_emb)
            new_node_emb = np.array(w[0][v1 + 1])
            new_w.append(new_node_emb)
            G = nx.from_numpy_matrix(self.m_train)
            for v1 in range(max_drug):
                if len(G[v1]) > 0:
                    new_node_emb = np.array(w[0][v1])
                    other_nodes_w = np.zeros(len(new_node_emb))  # empty vector
                    total_weights = 0
                    # print(other_nodes_w)
                    # w2 = 0
                    for v2 in G[v1]:
                        curr_weight = 1 / len(G[v1])  #
                        # w2+=len(G[v2])
                        total_weights += curr_weight
                        other_nodes_w += curr_weight * w[0][v2]
                    # w1 = len(G[v1])
                    # w2 /= len
                    new_node_emb = new_node_emb * orig_ratio + (
                            1 - orig_ratio) * other_nodes_w / total_weights  # here the orig_ratio is 1-alpha from the paper.
                    # new_w.append(new_node_emb*orig_ratio + (1-orig_ratio)*other_nodes_w/total_weights)
                else:
                    new_node_emb = np.array(w[0][v1])
                    # new_w.append(np.array(w[0][v1]))
                # old_w = new_w[v1]
                # new_w[v1] = new_node_emb
                # self.mult_dense.set_weights(np.array([np.array(new_w)]))
                new_w[v1] = new_node_emb
            #     w[0][v1] = new_node_emb
            #     self.mult_dense.set_weights(w)
            #     #y_pred = self.model_emb.predict(self.validation_features,batch_size=100000)
            #     loss = self.model_emb.evaluate(self.validation_features,self.validation_target_att, batch_size=100000,verbose=0)[0]
            #     #auc = roc_auc_score(self.validation_target_att, y_pred)
            #     print(f'Loss for {v1}, {loss }')
            #
            #     w[0][v1] = new_w[v1] #new_w now have the original weight
            #     self.mult_dense.set_weights(w)
            #     if loss < first_loss:
            #         new_w[v1] = new_node_emb
            #         replaced_count+=1
            # print(f'replaced embedding: {replaced_count}')
            self.text_embedding_layer2.set_weights(np.array([np.array(new_w)]))


    def update_text_weights(self,orig_ratio,iterations=1):

        self.text_embedding_layer.set_weights(self.tw)
        max_drug = len(self.i2d)
        print(max_drug)
        for x in range(iterations):
            # y_pred = self.model_emb.predict(self.validation_features, batch_size=100000)
            # first_loss = self.model_emb.evaluate(self.validation_features,self.validation_target_att, batch_size=100000,verbose=0)[0]
            # first_loss = roc_auc_score(self.validation_target_att, y_pred)
            w = self.text_embedding_layer.get_weights()
            new_w = []
            replaced_count = 0
            for v1 in range(max_drug):
                new_node_emb = np.array(w[0][v1])
                new_w.append(new_node_emb)
            new_node_emb = np.array(w[0][v1+1])
            new_w.append(new_node_emb)
            G = nx.from_numpy_matrix(self.m_train)
            for v1 in range(max_drug):
                if len(G[v1]) > 0:
                    new_node_emb = np.array(w[0][v1])
                    other_nodes_w = np.zeros(len(new_node_emb))  # empty vector
                    total_weights = 0
                    # print(other_nodes_w)
                    # w2 = 0
                    for v2 in G[v1]:
                        curr_weight = 1 / len(G[v1])  #
                        # w2+=len(G[v2])
                        total_weights += curr_weight
                        other_nodes_w += curr_weight * w[0][v2]
                    # w1 = len(G[v1])
                    # w2 /= len
                    new_node_emb = new_node_emb * orig_ratio + (
                                1 - orig_ratio) * other_nodes_w / total_weights  # here the orig_ratio is 1-alpha from the paper.
                    # new_w.append(new_node_emb*orig_ratio + (1-orig_ratio)*other_nodes_w/total_weights)
                else:
                    new_node_emb = np.array(w[0][v1])
                    # new_w.append(np.array(w[0][v1]))
                # old_w = new_w[v1]
                # new_w[v1] = new_node_emb
                # self.mult_dense.set_weights(np.array([np.array(new_w)]))
                new_w[v1] = new_node_emb
            #     w[0][v1] = new_node_emb
            #     self.mult_dense.set_weights(w)
            #     #y_pred = self.model_emb.predict(self.validation_features,batch_size=100000)
            #     loss = self.model_emb.evaluate(self.validation_features,self.validation_target_att, batch_size=100000,verbose=0)[0]
            #     #auc = roc_auc_score(self.validation_target_att, y_pred)
            #     print(f'Loss for {v1}, {loss }')
            #
            #     w[0][v1] = new_w[v1] #new_w now have the original weight
            #     self.mult_dense.set_weights(w)
            #     if loss < first_loss:
            #         new_w[v1] = new_node_emb
            #         replaced_count+=1
            # print(f'replaced embedding: {replaced_count}')
            self.text_embedding_layer.set_weights(np.array([np.array(new_w)]))

        return

    def set_concept_embedding_matrix(self):
        self.concept_embedding_matrix = np.zeros((len(self.i2d) + 1, self.text_embedding_size))

        idx = 0
        for item in self.i2d:
            drug_name = self.drug_id_to_name[item]
            res =  (self.drugs_concept_names.loc[self.drugs_concept_names['name'] == drug_name]).reset_index()
            if res.size > 0:
                concept_name = (res.loc[0])[CONCEPT_NAME_F]
                if not (isNaN(concept_name)):
                    if (CONCEPT_NAME_F == 'concept_mesh'):
                        self.concept_embedding_matrix[idx] = np.asarray(self.text_embedding_model[concept_name], dtype=np.float32)
                    elif (CONCEPT_NAME_F == 'name'):   #BERT
                        self.concept_embedding_matrix[idx] = np.asarray(self.text_embedding_model.loc[concept_name], dtype=np.float32)
                    else:
                        self.concept_embedding_matrix[idx] = self.text_embedding_model[concept_name]

                else:
                    concept_name = ""
                    #self.concept_embedding_matrix[idx] = self.default_text_emb  # we can initiate to zeros to begin with
            else:
                concept_name = ""
                #self.concept_embedding_matrix[idx] = self.default_text_emb   #we can initiate to zeros to begin with



            # if concept_name != "":
            #     self.concept_embedding_matrix[idx] = self.text_embedding_model[concept_name[0]]
            #     #("remove ",self.concept_embedding_matrix[idx])
            # else:
            #     self.concept_embedding_matrix[idx] = self.default_text_emb   #we can initiate to zeros to begin with
            idx = idx + 1



    def set_default_text_embedding_vec(self):
        first = True
        nitems = 0
        for index,row in self.drugs_concept_names.iterrows():
            if ((row[CONCEPT_NAME_F] is not None) and not (pd.isnull(row[CONCEPT_NAME_F]))):
                nitems = nitems + 1
                if USE_BERT:
                    val = np.asarray(self.text_embedding_model.loc[row[CONCEPT_NAME_F]], dtype=np.float32)
                else:
                    val = np.asarray(self.text_embedding_model[row[CONCEPT_NAME_F]], dtype=np.float32)
                if first:
                    vec = val
                    first = False
                else:
                    vec = vec + val
                    #vec = vec + np.asarray(self.text_embedding_model[row[CONCEPT_NAME_F]], dtype=np.float32)
        vec = vec / (1.0*nitems)
        return vec

    def load_embedding_concept(self,path):
        with open(path) as json_file:
            concept_vectors = json.load(json_file)
        return concept_vectors

    def load_embedding_bert(self,path):
            text_embeddings = pd.read_csv(path, index_col='name')
            text_embeddings = text_embeddings[[x for x in text_embeddings.columns if 'emb_' in x]]
            return text_embeddings

    def load_embedding(self,path, binary):
        #print("remove path ", path)
        embedding = KeyedVectors.load_word2vec_format(datapath(path), binary=True)
        #print('remove embedding loaded from', path)
        return embedding

    def save_model(self):
        print("saving model ",self.model_file_name)
        try:
            self.model_emb.save(self.model_file_name)
        except:
            print("error saving model ",self.model_file_name)

    def load_model(self):
        try:
            print("cwd ",os.getcwd())
            self.model_emb =  keras.models.load_model(self.model_file_name)
        except:
            print("error loading model ", self.model_file_name)



    def get_sample_train_validation(self, train_pos, train_neg, validation_pos, validation_neg, neg_to_pos_ratio=1.0):
        if neg_to_pos_ratio is None:
            train, validation = train_pos + train_neg, validation_pos + validation_neg
        else:
            train = list()
            # validation = list()
            # random.shuffle(validation_pos);random.shuffle(validation_neg)
            # random.shuffle(train_pos);random.shuffle(train_neg)
            #         train = sample_each_drug_once(train_pos,train)
            #         train = sample_each_drug_once(train_neg,train)
            #         validation = sample_each_drug_once(validation_pos,validation)
            #         validation = sample_each_drug_once(validation_neg,validation)
            train = list(train_pos)
            # validation = list(validation_pos)
            if len(train_pos) * neg_to_pos_ratio < len(train_neg):
                train += random.sample(train_neg, int(len(train_pos) * neg_to_pos_ratio))
            else:
                print('not sampling due to increased number of positive samples')
                train += train_neg
                # validation += random.sample(validation_neg, len(validation_pos))
            validation = validation_pos + validation_neg

        train = [(x[0], x[1]) if random.random() > 0.5 else (x[1], x[0]) for x in
                 train]  # this is redundent now as shared layer is used
        validation = [(x[0], x[1]) if random.random() > 0.5 else (x[1], x[0]) for x in validation]
        return train, validation

    def create_pos_neg_instances(self,train_tuples, validation_tuples, m_train,m_validation):
        train_pos = [x for x in train_tuples if m_train[x[0], x[1]] == 1]
        train_neg = [x for x in train_tuples if m_train[x[0], x[1]] == 0]
        validation_pos = [x for x in validation_tuples if m_validation [x[0], x[1]] == 1]
        validation_neg = [x for x in validation_tuples if m_validation [x[0], x[1]] == 0]
        print(
            f'train pos: {len(train_pos)}, train neg: {len(train_neg)}, val pos: {len(validation_pos)}, val neg: {len(validation_neg)}')
        return train_pos, train_neg, validation_pos, validation_neg

    def fit(self):
        self.init_nn_model()
        self.fit_nn_model()


    def get_embeddings(self):
        return [np.concatenate([a, b]) for a,b in zip(self.mult_dense.get_weights()[0],self.mlp.get_weights()[0])]

    def get_instances(self, tuples_sample, m):
        instance_features = []
        instance_features.append(np.array([t[0] for t in tuples_sample]))
        instance_features.append(np.array([t[1] for t in tuples_sample]))
        # array_emb_0 = np.array()
        # array_emb_1 = np.array()
        # for t in tuples_sample:
        #     drug_name = self.drug_id_to_name[t[0]]
        #     concept_name = self.drugs_concept_names.loc[self.drugs_concept_names['name'] == drug_name, 'concept_name']
        #     if concept_name is not None:
        #         con_emb = self.text_embedding_model[concept_name]
        #     else:
        #         con_emb = self.default_text_emb
        #     array_emb_0 = np.append(array_emb_0,con_emb)
        #     drug_name = self.drug_id_to_name[t[1]]
        #     concept_name = self.drugs_concept_names.loc[self.drugs_concept_names['name'] == drug_name, 'concept_name']
        #     if concept_name is not None:
        #         con_emb = self.text_embedding_model[concept_name]
        #     else:
        #         con_emb = self.default_text_emb
        #     array_emb_1 = np.append(array_emb_1,con_emb)
        #     instance_features.append(array_emb_0)
        #     instance_features.append(array_emb_1)


        target_att = np.array([[m[t[0], t[1]]] for t in tuples_sample])
        return instance_features, target_att


    def get_instances_backup(self, tuples_sample, m):
        instance_features = []
        instance_features.append(np.array([t[0] for t in tuples_sample]))
        instance_features.append(np.array([t[1] for t in tuples_sample]))
        array_emb_0 = np.array()
        array_emb_1 = np.array()
        for t in tuples_sample:
            drug_name = self.drug_id_to_name[t[0]]
            #concept_name = self.drugs_concept_names.loc[self.drugs_concept_names['name'] ==
            concept_name = self.drugs_concept_names.loc[self.drugs_concept_names['name'] == drug_name, CONCEPT_NAME_F]
            if concept_name is not None:
                if (CONCEPT_NAME_F == 'concept_mesh'):
                    con_emb = np.asarray(self.text_embedding_model[concept_name], dtype=np.float32)
                elif (CONCEPT_NAME_F == 'name'): #BERT
                    con_emb = np.asarray(self.text_embedding_model.loc[concept_name],dtype=np.float32)
                else:
                    con_emb = self.text_embedding_model[concept_name]
            else:
                con_emb = self.default_text_emb
            array_emb_0 = np.append(array_emb_0,con_emb)
            drug_name = self.drug_id_to_name[t[1]]
            # = self.drugs_concept_names.loc[self.drugs_concept_names['name'] == drug_name, 'concept_name']
            concept_name = self.drugs_concept_names.loc[self.drugs_concept_names['name'] == drug_name, CONCEPT_NAME_F]
            if concept_name is not None:
                if (CONCEPT_NAME_F == 'concept_mesh'):
                    con_emb = np.asarray(self.text_embedding_model[concept_name], dtype=np.float32)
                elif (CONCEPT_NAME_F == 'name'):
                    con_emb = np.asarray(self.text_embedding_model.loc[concept_name],dtype=np.float32)
                else:
                    con_emb = self.text_embedding_model[concept_name]
            else:
                con_emb = self.default_text_emb
            array_emb_1 = np.append(array_emb_1,con_emb)
            instance_features.append(array_emb_0)
            instance_features.append(array_emb_1)


        target_att = np.array([[m[t[0], t[1]]] for t in tuples_sample])
        return instance_features, target_att


    def init_nn_model_old(self):
        print("start init ")
        input_node_a = Input(shape=(1,), name='b')
        input_node_b = Input(shape=(1,), name='c')
        input_node_c = Input(shape=(self.text_embedding_size,),name='te_b')
        input_node_d = Input(shape=(self.text_embedding_size,),name='te_c')


        regularization = 0
        mlp_emb = Embedding(output_dim=1, name='MLP_embedding',input_dim=self.num_of_drugs,embeddings_regularizer=l2(regularization))
        self.mlp = mlp_emb
        emb_mlp1 = mlp_emb(input_node_a)
        emb_mlp2 = mlp_emb(input_node_b)
        l_mlp = Add()([emb_mlp1,emb_mlp2])

        mult_dense = Embedding(output_dim=self.mul_emb_size, name='GMF_embedding',embeddings_regularizer=l2(regularization),input_dim=self.num_of_drugs) #
        self.mult_dense = mult_dense
        emb_mult1 = mult_dense(input_node_a)
        emb_mult2 = mult_dense(input_node_b)
        dr_emb_mult1 = Dropout(self.dropout)(emb_mult1)
        dr_emb_mult2 = Dropout(self.dropout)(emb_mult2)
        mult = Dropout(0)(Multiply()([dr_emb_mult1, dr_emb_mult2]))
        num1 = Flatten()(l_mlp)
        num2 = Flatten()(mult)
        num3= Flatten()(input_node_c)
        num4 = Flatten()(input_node_d)
        #final_layers = Concatenate(axis=-1)([x for i, x in enumerate([Flatten()(l_mlp), Flatten()(mult),Flatten()(input_node_c),Flatten()(input_node_d)])])
        final_layers = Concatenate(axis=-1)([x for i, x in enumerate([num1,num2,num3,num4])])
        main_output= Dense(1, activation='sigmoid')(final_layers)#the init is critical for the model to work
        model_emb = Model(inputs=[input_node_a,input_node_b], outputs=main_output)  # fixed_input
        model_emb.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['mae'])  # binary_crossentropy

        #from keras.utils.vis_utils import plot_model
        #import os
        #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        #plot_model(model_emb, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # self.model_gmf = Model(inputs=[input_node_a,input_node_b], outputs=mult_output) #fixed_input
        # self.model_gmf.compile(optimizer=Adam(),loss='binary_crossentropy', metrics=['accuracy','mae']) #binary_crossentropy
        #
        # self.model_mlp = Model(inputs=[input_node_a,input_node_b], outputs=n_con_output) #fixed_input
        # self.model_mlp.compile(optimizer=Adam(),loss='binary_crossentropy', metrics=['accuracy','mae']) #binary_crossentropy

        print("end init")
        self.model_emb = model_emb

    def init_combiner_model(self):

        # inputs = Input(shape=(2,))
        # output = Dense(1, activation='sigmoid',use_bias=True)(inputs)
        # logistic_model = Model(inputs, output)
        #
        # # Compile the model
        # logistic_model.compile(optimizer='sgd',
        #                loss='binary_crossentropy',
        #                metrics=['mae'])
        # self.combiner_model = logistic_model

        n_layers = 9

        inputs = Input(shape=(2,))
        x = Dense(200, activation='relu',use_bias=True)(inputs)
        x = Dropout(0.4)(x)
        for layer in range(n_layers - 1):
            x = Dense(200, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        deep_n_net = Model(inputs, output)

        deep_n_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
        self.combiner_model = deep_n_net

    def init_nn_model0(self):

        print("start init ", "num of drugs ", self.num_of_drugs, " i2d len ", len(self.i2d))


        input_node_a = Input(shape=(1,), name='b')
        input_node_b = Input(shape=(1,), name='c')
        regularization = 0
        text_embedding_layer = Embedding(input_dim=len(self.i2d) + 1,
                                 output_dim=self.text_embedding_size,
                                 weights=[self.concept_embedding_matrix],
                                 trainable=False, name='Text_embedding')
        self.text_embedding_layer = text_embedding_layer
        emb_text1 = text_embedding_layer(input_node_a)

        emb_text2 = text_embedding_layer(input_node_b)
        dr_emb_text2 = Dropout(self.dropout)(emb_text2)
        dr_emb_text1 = Dropout(self.dropout)(emb_text1)

        print("remove after emb 222")
        mult_text = Concatenate()([emb_text1, emb_text2])
        # mult_text = Dot(axes=1)([dr_emb_text1, dr_emb_text2])
        # num3 = Flatten()(mult_text)
        ##text_relu = Dropout(self.dropout)(Dense(70, activation='relu')(mult_text))
        text_relu = Dense(70, activation='relu')(mult_text)
        ##final_layers = Flatten()(text_relu)
        print("before num3i")
        num3i = Flatten()(text_relu)
        print("after num3i")
        final_text = Dense(1, activation='sigmoid', name='text_output')(num3i)
        model_text = Model(inputs=[input_node_a, input_node_b], outputs=final_text)  # fixed_input
        model_text.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['mae'])  # binary_crossentropy
        print(model_text.summary())
        self.model_text = model_text


    def init_nn_model(self):
        self.init_nn_model0()
        self.init_combiner_model()
        print("start init ","num of drugs ",self.num_of_drugs," i2d len ",len(self.i2d))
        input_node_a = Input(shape=(1,), name='b')
        input_node_b = Input(shape=(1,), name='c')
        #input_node_at = Input(shape=(1,), name='bt')
        #input_node_bt = Input(shape=(1,), name='ct')

        regularization = 0
        mlp_emb = Embedding(output_dim=1, name='MLP_embedding',input_dim=self.num_of_drugs,embeddings_regularizer=l2(regularization))
        self.mlp = mlp_emb
        emb_mlp1 = mlp_emb(input_node_a)
        emb_mlp2 = mlp_emb(input_node_b)
        l_mlp = Add()([emb_mlp1,emb_mlp2])

        mult_dense = Embedding(output_dim=self.mul_emb_size, name='GMF_embedding',embeddings_regularizer=l2(regularization),input_dim=self.num_of_drugs) #
        self.mult_dense = mult_dense
        emb_mult1 = mult_dense(input_node_a)
        emb_mult2 = mult_dense(input_node_b)
        dr_emb_mult1 = Dropout(self.dropout)(emb_mult1)
        dr_emb_mult2 = Dropout(self.dropout)(emb_mult2)

        text_embedding_layerc = Embedding(input_dim=len(self.i2d) + 1,
                                         output_dim=self.text_embedding_size,
                                         weights=[self.concept_embedding_matrix],
                                         trainable=False, name='Text_embedding')
        self.text_embedding_layer2 = text_embedding_layerc
        print("remove after emb 1")
        emb_text1 = text_embedding_layerc(input_node_a)

        emb_text2 = text_embedding_layerc(input_node_b)
        dr_emb_text2 = Dropout(self.dropout)(emb_text2)
        dr_emb_text1 = Dropout(self.dropout)(emb_text1)
        print("remove after emb 2")
        dr_emb_teint1 = Concatenate()([dr_emb_mult1,dr_emb_text1])
        dr_emb_teint2 = Concatenate()([dr_emb_mult2, dr_emb_text2])

        mult = Dropout(0)(Multiply()([dr_emb_mult1, dr_emb_mult2]))
        #mult = Dropout(0)(Multiply()([dr_emb_teint1, dr_emb_teint2]))
        num1 = Flatten()(l_mlp)
        num2 = Flatten()(mult)
        print("remove after emb 3")


        ##mult_text = Dropout(self.dropout)(Multiply()([dr_emb_text1, dr_emb_text2]))
        ##mult_text = Multiply()([emb_text1, emb_text2])
        mult_text = Concatenate()([emb_text1, emb_text2])
        #mult_text = Dot(axes=1)([dr_emb_text1, dr_emb_text2])
        ##num3 = Flatten()(mult_text)
        text_relu = Dropout(self.dropout)(Dense(70, activation='relu')(mult_text))
        #text_relu = Dense(70, activation='relu')(mult_text)
        ##final_layers = Flatten()(text_relu)
        ##print("before num3i")
        num3i = Flatten()(text_relu)
        ##print("after num3i")
        ##final_text =  Dense(1, activation='sigmoid',name='text_output')(num3i)
        # model_text = Model(inputs=[input_node_a,input_node_b], outputs=final_text)  # fixed_input
        # model_text.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['mae'])  # binary_crossentropy
        ##input_model_base = self.model_text.input
        # self.model_text = model_text
        ##print("after num3ii")
        ##final_layers = Flatten()(mult_text)
        ##num3 = Flatten()(emb_text1)
        ##num4 = Flatten()(emb_text2)
        # print("remove after flatten")
        #base_output = self.model_text.get_layer('text_output').output
        base_output = self.model_text([input_node_a,input_node_b])
        # print("remove after flatting2")
        #num3 = Flatten()(base_output)
        #num3 = base_output

        ##final_layers = Flatten()(text_relu)
        #final_layers = Concatenate(axis=-1)([x for i, x in enumerate([Flatten()(l_mlp), Flatten()(mult),Flatten()(input_node_c),Flatten()(input_node_d)])])
        ##final_layers = Concatenate(axis=-1)([x for i, x in enumerate([num1,num2,num3i])])   #combined good
        final_layers = Concatenate(axis=-1)([x for i, x in enumerate([num1,num2])])    #no text
        ##final_layers = Concatenate(axis=-1)([x for i, x in enumerate([num3])])   #only text
        ##final_layers = num3i
        # if ONLY_TEXT:
        #     main_output2 = base_output
        # elif NO_TEXT:
        #     main_output2 = Dense(1, activation='sigmoid')(final_layers)  # the init is critical for the model to work
        # else:
        main_output= Dense(1, activation='sigmoid')(final_layers)
        #main_output2 = Dense(1, activation='sigmoid')(Concatenate()([main_output,base_output]))
        print("remove after concat 2")
        model_emb = Model(inputs=[input_node_a,input_node_b], outputs=main_output)  # fixed_input
        print("remove before compile")
        model_emb.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['mae'])  # binary_crossentropy

        #from keras.utils.vis_utils import plot_model
        #import os
        #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        #plot_model(model_emb, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # self.model_gmf = Model(inputs=[input_node_a,input_node_b], outputs=mult_output) #fixed_input
        # self.model_gmf.compile(optimizer=Adam(),loss='binary_crossentropy', metrics=['accuracy','mae']) #binary_crossentropy
        #
        # self.model_mlp = Model(inputs=[input_node_a,input_node_b], outputs=n_con_output) #fixed_input
        # self.model_mlp.compile(optimizer=Adam(),loss='binary_crossentropy', metrics=['accuracy','mae']) #binary_crossentropy

        print("end init")
        print(model_emb.summary())
        self.model_emb = model_emb


    def fit_nn_model(self):

        print("start fit_nn")

        learning_rate = self.learning_rate
        batch_size = self.batch_size
        epoch_combiner = 15
        epochs=self.epochs
        epochs_text = self.text_epochs
        train_tuples, validation_tuples,self.m_train = create_train_validation_split(self.m, train_ratio=1)  # if ratio = 1 then no validation
        #train_tuples, validation_tuples, self.m_train = create_train_validation_split_single_sample_per_drug(self.m,train_ratio=0.75)


        train_pos, train_neg, validation_pos, validation_neg = self.create_pos_neg_instances(train_tuples, validation_tuples,self.m_train, self.m)

        cnt_epoch=0
        current_learning_rate = learning_rate

        checkpoint_filepath = 'F:\\Drugs\\Code\\DDI_prediciton_mm_v3\\multimodal_learning\\output\\xx1.mdl_wts.hdf5'
        checkpoint_filepath2 = 'F:\\Drugs\\Code\\DDI_prediciton_mm_v3\\multimodal_learning\\output\\mymodel'

        self.checkpoint_filepath = checkpoint_filepath
        self.checkpoint_filepath2 = checkpoint_filepath2
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='max',
            save_best_only=True)

        K.set_value(self.model_emb.optimizer.lr, learning_rate)
        # create sample instances#
        train_tuples_sample, validation_tuples_sample = self.get_sample_train_validation(train_pos, train_neg,
                                                                                         validation_pos, validation_neg,
                                                                                         neg_to_pos_ratio=self.neg_per_pos_sample)
        train_features, train_target_att = self.get_instances(train_tuples_sample, self.m_train)
        if self.validation_features == None:
            self.validation_features, self.validation_target_att = self.get_instances(validation_tuples_sample,
                                                                                      self.m, )

        # if self.validation_features != None and len(self.validation_target_att) > 0:
        #     print('before fit text v2')
        #     self.model_text.fit(x=train_features, y=train_target_att, batch_size=batch_size,
        #                         callbacks=[model_checkpoint_callback], epochs=epochs_text, verbose=2,
        #                         validation_data=(self.validation_features, self.validation_target_att))  # ,callbacks=
        # else:
        #     print("before fit text v")
        #     self.model_text.fit(x=train_features, y=train_target_att, batch_size=batch_size, epochs=epochs_text,
        #                         verbose=2, callbacks=[
        #             model_checkpoint_callback])  # ,validation_data=(self.validation_features, self.validation_target_att))  # ,callbacks=[earlycurrent_learning_rateop]

        # The model weights (that are considered the best) are loaded into the model.
        # self.model_text.load_weights


        ###########################
        ### Text stand alone model
        ############################

        while epochs_text > cnt_epoch:
            cnt_epoch+=1
            print(f"Epoch number {cnt_epoch} with LR  {current_learning_rate}")
            K.set_value(self.model_emb.optimizer.lr, learning_rate)
            # create sample instances#
            train_tuples_sample, validation_tuples_sample = self.get_sample_train_validation(train_pos, train_neg,
                                                                                             validation_pos, validation_neg,
                                                                                             neg_to_pos_ratio=self.neg_per_pos_sample)
            train_features, train_target_att = self.get_instances(train_tuples_sample,self.m_train)
            if self.validation_features==None:
                self.validation_features, self.validation_target_att = self.get_instances(validation_tuples_sample,self.m,)

            if self.validation_features!=None and len(self.validation_target_att)>0:
                print('before fit text v2')
                self.model_text.fit(x=train_features, y=train_target_att, batch_size=batch_size, callbacks=[model_checkpoint_callback], epochs=1,verbose=2,validation_data=(self.validation_features, self.validation_target_att) ) # ,callbacks=
            else:
                print("before fit text v22")
                self.model_text.fit(x=train_features, y=train_target_att, batch_size=batch_size, epochs=1,
                                  verbose=2,callbacks=[model_checkpoint_callback])  # ,validation_data=(self.validation_features, self.validation_target_att))  # ,callbacks=[earlycurrent_learning_rateop]
        self.tw = self.text_embedding_layer.get_weights()
        if sum(self.validation_target_att) > 0:
            y_pred = self.model_text.predict(self.validation_features)
            auc = roc_auc_score(self.validation_target_att, y_pred)
            best_auc = None
            best_x = None
            print(
                f"new evalm before text: {auc}, {self.model_text.evaluate(self.validation_features, self.validation_target_att, verbose=0)}")
            results = []
            for i in range(1, 1 + 1):
                for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    self.update_text_weights(orig_ratio=x, iterations=i)
                    y_pred = self.model_text.predict(self.validation_features, batch_size=50000)
                    auc = roc_auc_score(self.validation_target_att, y_pred)
                    loss = self.model_text.evaluate(self.validation_features, self.validation_target_att, verbose=0,
                                                    batch_size=50000)[0]
                    print(f"new evalm text 2 {x}, {i}: {auc}, {loss}")
                    results.append((x, i, auc, loss))
                    if best_auc == None or auc > best_auc:
                        best_auc = auc
                        best_x = x
            print(f'best propagation AUC text {best_auc}, best factpr: {best_x}')
            print("text results=", results)
        print('text AFMFP 123 ',self.text_propgation_factor)
        if self.text_propgation_factor != None:
            if self.text_propgation_factor > 0.0:
                print(f'setting text weights to {self.text_propgation_factor}')
                # print(f'setting text weights to {best_x}')
                self.update_text_weights(orig_ratio=self.text_propgation_factor)
        print("DONE text!")



        # The model weights (that are considered the best) are loaded into the model.
        ##self.model_text.load_weights(checkpoint_filepath)
        self.model_text.save(checkpoint_filepath2)

        ########################################################
        ### Interaction model
        ########################################################

        cnt_epoch=0
        current_learning_rate = learning_rate
        while epochs > cnt_epoch:
            cnt_epoch+=1
            print(f"Epoch number {cnt_epoch} with LR {current_learning_rate}")
            K.set_value(self.model_emb.optimizer.lr, learning_rate)
            # create sample instances#
            train_tuples_sample, validation_tuples_sample = self.get_sample_train_validation(train_pos, train_neg,
                                                                                             validation_pos, validation_neg,
                                                                                             neg_to_pos_ratio=self.neg_per_pos_sample)
            train_features, train_target_att = self.get_instances(train_tuples_sample,self.m_train)
            if self.validation_features==None:
                self.validation_features, self.validation_target_att = self.get_instances(validation_tuples_sample,self.m,)

            if self.validation_features!=None and len(self.validation_target_att)>0:
                print('before fit emb')
                self.model_emb.fit(x=train_features, y=train_target_att, batch_size=batch_size, epochs=1,verbose=2,validation_data=(self.validation_features, self.validation_target_att) ) # ,callbacks=[earlycurrent_learning_rateop]
                y_pred = self.model_emb.predict(self.validation_features,batch_size=50000)
                auc = roc_auc_score(self.validation_target_att, y_pred)
                print(f'auc: {auc}')
            else:
                print("before model emb fit v")
                self.model_emb.fit(x=train_features, y=train_target_att, batch_size=batch_size, epochs=1,
                                 verbose=2)  # ,validation_data=(self.validation_features, self.validation_target_att))  # ,callbacks=[earlycurrent_learning_rateop]

        self.w = self.mult_dense.get_weights()
        if sum(self.validation_target_att) > 0:
            y_pred = self.model_emb.predict(self.validation_features)
            auc = roc_auc_score(self.validation_target_att, y_pred)
            best_auc = None
            best_x = None
            print(
                f"new evalm before: {auc}, {self.model_emb.evaluate(self.validation_features, self.validation_target_att, verbose=0)}")
            results = []
            for i in range(1, 1 + 1):
                for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    self.update_weights(orig_ratio=x, iterations=i)
                    y_pred = self.model_emb.predict(self.validation_features, batch_size=50000)
                    auc = roc_auc_score(self.validation_target_att, y_pred)
                    loss = self.model_emb.evaluate(self.validation_features, self.validation_target_att, verbose=0,
                                                   batch_size=50000)[0]
                    print(f"new evalm {x}, {i}: {auc}, {loss}")
                    results.append((x, i, auc, loss))
                    if best_auc == None or auc > best_auc:
                        best_auc = auc
                        best_x = x
            print(f'best propagation AUC {best_auc}, best factpr: {best_x}')
            print("results=", results)

        if self.propgation_factor != None:
            print(f'setting weights to {self.propgation_factor}')
            self.update_weights(orig_ratio=self.propgation_factor)
        print("DONE!")
        print('new version 2')

        self.tw2 = self.text_embedding_layer2.get_weights()
        if sum(self.validation_target_att)>0:
            print('remove here 2')
            y_pred = self.model_emb.predict(self.validation_features)
            auc = roc_auc_score(self.validation_target_att, y_pred)
            best_auc = None
            best_x = None
            print(
                f"new evalm before text2: {auc}, {self.model_emb.evaluate(self.validation_features,self.validation_target_att,verbose=0)}")
            results = []
            for i in range(1, 1 + 1):
                for x in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    self.update_text_weights2(orig_ratio=x, iterations=i)
                    y_pred = self.model_emb.predict(self.validation_features, batch_size=50000)
                    auc = roc_auc_score(self.validation_target_att, y_pred)
                    loss = self.model_emb.evaluate(self.validation_features, self.validation_target_att, verbose=0,
                                                    batch_size=50000)[0]
                    print(f"new evalm text 2 {x}, {i}: {auc}, {loss}")
                    results.append((x, i, auc, loss))
                    if best_auc == None or auc > best_auc:
                        best_auc = auc
                        best_x = x
            print(f'best propagation AUC 2 {best_auc}, best factpr: {best_x}')
            print("text results=", results)

        print('before 222 remove')
        if self.text_propgation_factor2 != None:
            if self.text_propgation_factor2 > 0:
                print('???? remove')
            # print(f'setting text weights to {self.propgation_factor}')
                print(f'setting text weights 2 to {self.text_propgation_factor2}')
                self.update_text_weights2(orig_ratio=self.text_propgation_factor2)
        print("DONE text 2!")


        ######################################################
        ###### Combiner model
        ######################################################

        cnt_epoch = 0
        while epoch_combiner > cnt_epoch:
            cnt_epoch+=1
            print(f"Epoch number {cnt_epoch} with LR {current_learning_rate}")
            K.set_value(self.model_emb.optimizer.lr, learning_rate)
            # create sample instances#
            train_tuples_sample, validation_tuples_sample = self.get_sample_train_validation(train_pos, train_neg,
                                                                                             validation_pos, validation_neg,
                                                                                             neg_to_pos_ratio=self.neg_per_pos_sample)
            train_features, train_target_att = self.get_instances(train_tuples_sample,self.m_train)
            if self.validation_features==None:
                self.validation_features, self.validation_target_att = self.get_instances(validation_tuples_sample,self.m,)

            if self.validation_features!=None and len(self.validation_target_att)>0:
                #print('before fit comb v2 c')
                #print(type(train_features))
                text_pred = self.model_text.predict(train_features)
                emb_pred = self.model_emb.predict(train_features)
                #print('remove fit comb 2')
                text_pred_val = self.model_text.predict(self.validation_features)
                emb_pred_val = self.model_emb.predict(self.validation_features)
                text_pred_val = text_pred_val[:,0]
                emb_pred_val = emb_pred_val[:,0]
                emb_pred = emb_pred[:,0]
                text_pred = text_pred[:,0]
                #print('remove fit comb 2.5',type(text_pred),type(emb_pred),type(text_pred_val),type(emb_pred_val))
                x_dat = np.array([text_pred,emb_pred]).transpose()
                #print('remove fit comb 2.8',text_pred.shape,emb_pred.shape,text_pred_val.shape,emb_pred_val.shape)
                x_dat_val = np.array([text_pred_val,emb_pred_val]).transpose()
                #print('remove fit comb 3',type(x_dat),type(train_target_att),type(x_dat_val),type(self.validation_target_att))
                #print(x_dat.shape,train_target_att.shape,x_dat_val.shape,self.validation_target_att.shape)
                self.combiner_model.fit(x=x_dat, y=train_target_att[:,0], batch_size=batch_size,epochs=1,verbose=2,validation_data=(x_dat_val, self.validation_target_att[:,0]) ) # ,callbacks=
                #print('after fit 4')
            else:
                print("before fit comb v")
                text_pred = self.model_text.predict(train_features)
                emb_pred = self.model_emb.predict(train_features)
                emb_pred = emb_pred[:,0]
                text_pred = text_pred[:,0]
                #print('remove fit comb 2.5',type(text_pred),type(emb_pred),type(text_pred_val),type(emb_pred_val))
                x_dat = np.array([text_pred,emb_pred]).transpose()

                #text_pred = self.clean_text_predict(x=train_features)
                #emb_pred = self.clean_emb_predict(x=train_features)
                #x_dat = np.array([text_pred,emb_pred])
                self.combiner_model.fit(x=x_dat, y=train_target_att[:,0], batch_size=batch_size, epochs=1, verbose=2)  # ,callbacks=



            # The model weights (that are considered the best) are loaded into the model.
            ##self.model_text.load_weights(checkpoint_filepath)
            self.model_text.save(checkpoint_filepath2)




        print("end fit nn new ")

    def update_weights(self,orig_ratio,iterations=1):
        self.mult_dense.set_weights(self.w)

        for x in range(iterations):
            #y_pred = self.model_emb.predict(self.validation_features, batch_size=100000)
            #first_loss = self.model_emb.evaluate(self.validation_features,self.validation_target_att, batch_size=100000,verbose=0)[0]
            #first_loss = roc_auc_score(self.validation_target_att, y_pred)
            w = self.mult_dense.get_weights()
            new_w =[]
            replaced_count = 0
            for v1 in range(self.num_of_drugs):

                new_node_emb = np.array(w[0][v1])
                new_w.append(new_node_emb)

            G = nx.from_numpy_matrix(self.m_train)
            for v1 in range(self.num_of_drugs):
                if len(G[v1])>0:
                    new_node_emb = np.array(w[0][v1])
                    other_nodes_w = np.zeros(len(new_node_emb)) #empty vector
                    total_weights=0
                    #print(other_nodes_w)
                    #w2 = 0
                    for v2 in G[v1]:
                        curr_weight = 1/len(G[v1])#
                        #w2+=len(G[v2])
                        total_weights+=curr_weight
                        other_nodes_w += curr_weight *w[0][v2]
                    #w1 = len(G[v1])
                    #w2 /= len(G[v1])
                    new_node_emb = new_node_emb*orig_ratio + (1-orig_ratio)*other_nodes_w/total_weights #   here the orig_ratio is 1-alpha from the paper.
                    #new_w.append(new_node_emb*orig_ratio + (1-orig_ratio)*other_nodes_w/total_weights)
                else:
                    new_node_emb = np.array(w[0][v1])
                    #new_w.append(np.array(w[0][v1]))
                #old_w = new_w[v1]
                #new_w[v1] = new_node_emb
                #self.mult_dense.set_weights(np.array([np.array(new_w)]))
                new_w[v1] = new_node_emb
            #     w[0][v1] = new_node_emb
            #     self.mult_dense.set_weights(w)
            #     #y_pred = self.model_emb.predict(self.validation_features,batch_size=100000)
            #     loss = self.model_emb.evaluate(self.validation_features,self.validation_target_att, batch_size=100000,verbose=0)[0]
            #     #auc = roc_auc_score(self.validation_target_att, y_pred)
            #     print(f'Loss for {v1}, {loss }')
            #
            #     w[0][v1] = new_w[v1] #new_w now have the original weight
            #     self.mult_dense.set_weights(w)
            #     if loss < first_loss:
            #         new_w[v1] = new_node_emb
            #         replaced_count+=1
            # print(f'replaced embedding: {replaced_count}')
            self.mult_dense.set_weights(np.array([np.array(new_w)]))

        # for x in range(iterations):
        #     w = self.mlp.get_weights()
        #     new_w =[]
        #     G = nx.from_numpy_matrix(self.m_train)
        #     for v1 in range(self.num_of_drugs):
        #         if len(G[v1])>0:
        #             new_node_emb = np.array(w[0][v1]) * orig_ratio
        #             for v2 in G[v1]:
        #                 new_node_emb += (1-orig_ratio)*(1/len(G[v1]))*w[0][v2]
        #             new_w.append(new_node_emb)
        #         else:
        #             new_w.append(np.array(w[0][v1]))
        #     self.mlp.set_weights(np.array([np.array(new_w)]))

            # for x in range(iterations):
            #     w = self.mult_bias.get_weights()
            #     new_w = []
            #     G = nx.from_numpy_matrix(self.m_train)
            #     for v1 in range(self.num_of_drugs):
            #         if len(G[v1]) > 0:
            #             new_node_emb = np.array(w[0][v1]) * orig_ratio
            #             for v2 in G[v1]:
            #                 new_node_emb += (1 - orig_ratio) * (1 / len(G[v1])) * w[0][v2]
            #             new_w.append(new_node_emb)
            #         else:
            #             new_w.append(np.array(w[0][v1]))
            #     self.mult_bias.set_weights(np.array([np.array(new_w)]))

            #print('bias values:',str(self.mult_bias.get_weights()))

    def clean_text_predict(self,x):
        batch_size = 100000
        eval_instances, _ = self.get_instances(x, self.m)
        #here we get (eval_instances) is a list of array for t[0] and t[1]
        print('remove clean text predict ',type(eval_instances),len(eval_instances))
        preds = self.model_text.predict(eval_instances,batch_size)
        print('remove clean text preds ',type(preds),preds.shape)
        return preds

    def clean_emb_predict(self,x):
        batch_size = 100000
        eval_instances, _ = self.get_instances(x, self.m)
        preds = self.model_emb.predict(eval_instances, batch_size)
        return preds


    def predict(self):

        # import math
        # import train_generator
        # importlib.reload(train_generator)
        # from train_generator import d2d_generators

        # d2d_generators_object = d2d_generators(m_train,train_tuples,validation_tuples)

        batch_size = 100000
        # m_evaluation = m_train
        # evaluation_tuples = validation_tuples
        m_evaluation = self.m
        evaluation_tuples = self.test_tuples  # test_tuples
        #evaluation_tuples = [(x,y) for y in range(self.num_of_drugs) for x in range(y+1,self.num_of_drugs)]#self.test_tuples  # test_tuples
        print(f'evaluating {len(evaluation_tuples)} instances')
        eval_instances, _ = self.get_instances(evaluation_tuples,self.m)
        # preds_him = get_pred_him()
        print('done creating instances')
        preds = self.model_emb.predict(eval_instances,
                                  batch_size=batch_size)  # [preds_him[x[0],x[1]] for x in evaluation_tuples] #

        if RANDOM_BASELINE:
            print("remove random")
            for i,val in enumerate(preds):
                rnd = random.random()
                if rnd >= RANDOM_POSITIVE_RATIO:
                    preds[i] = 1.0
                else:
                    preds[i] = 0.0
        # preds = self.model_text.predict(eval_instances,
        #                           batch_size=batch_size)  # [preds_him[x[0],x[1]] for x in evaluation_tuples] #

        print('done predicting', len(preds))
        count = 0
        predictions = np.zeros((self.m.shape[0], self.m.shape[1]))
        for i in range(len(evaluation_tuples)):
            idx1 = evaluation_tuples[i][0]
            idx2 = evaluation_tuples[i][1]
            predictions[idx1, idx2] = preds[i][0]
            count += 1
        eval_instances, _ = self.get_instances([(x[1], x[0]) for x in evaluation_tuples],self.m)

        preds = self.model_emb.predict(eval_instances,
                                  batch_size=batch_size)  # [preds_him[x[1],x[0]] for x in evaluation_tuples]#
        print("after2 remove type(preds) type(eval_instances) ",type(preds),type(eval_instances))

        # preds = self.model_text.predict(eval_instances,
        #                           batch_size=batch_size)  # [preds_him[x[1],x[0]] for x in evaluation_tuples]#


        if RANDOM_BASELINE:
            print("remove random")
            for i,val in enumerate(preds):
                rnd = random.random()
                if rnd >= RANDOM_POSITIVE_RATIO:
                    preds[i] = 1.0
                else:
                    preds[i] = 0.0
        for i in range(len(evaluation_tuples)):
            idx1 = evaluation_tuples[i][0]
            idx2 = evaluation_tuples[i][1]
            predictions[idx2, idx1] = preds[i][0]  # must be reversed...
            count += 1
        for i in range(self.m.shape[0]):
            for j in range(i + 1, self.m.shape[0]):
                from scipy import stats
                # new_score = stats.hmean([ max(0.000001,predictions[i,j]),max(0.000001,predictions[j,i]) ])
                new_score = (predictions[i, j] + predictions[j, i]) / 2
                # new_score = max(predictions[i,j],predictions[j,i])
                # new_score = predictions[j,i]
                predictions[i, j] = new_score
                predictions[j, i] = new_score
        for i in range(self.m.shape[0]):
            assert predictions[i, i] == 0



        # predictions2 = np.zeros((self.m.shape[0], self.m.shape[1]))
        # G = nx.from_numpy_matrix(self.m)
        # for v1,v2 in  [(i, j) for i in G.nodes() for j in G.nodes() if j>i]:
        #     nn_preds1 = []
        #     for z in G[v2]:
        #         nn_preds1.append(predictions[v1, z])
        #     nn_preds2 = []
        #     for z in G[v1]:
        #         nn_preds2.append(predictions[v2, z])
        #     predictions2[v1, v2] = 0.5*predictions[v1, v2] + 0.5*np.mean(np.nan_to_num([np.mean(nn_preds1),np.mean(nn_preds2)]))
        #     predictions2[v2, v1] = predictions2[v1, v2]

        #predictions = predictions2

        self.predicted_mat = predictions
        print('predicted: ', count)
        s = set([(x[0],x[1])for x in self.test_tuples])
        predictions = [(i, v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(predictions)],
                                                   reverse=True)]  # get the cells of matrix in ascending order of cell value
        print(1)
        predictions = [(t, v) for t, v in predictions
                       if (t[0], t[1]) in s ]  # just half of the matrix and predictions larger than 0
        if self.predictions_pickle_file_name!=None:
            nn_predicted_mat = self.predicted_mat
            pickling_on = open(self.predictions_pickle_file_name, "wb")
            pickle.dump(nn_predicted_mat, pickling_on)
            pickling_on.close()

        return predictions

    def predict_text(self):

        # import math
        # import train_generator
        # importlib.reload(train_generator)
        # from train_generator import d2d_generators

        # d2d_generators_object = d2d_generators(m_train,train_tuples,validation_tuples)

        batch_size = 100000
        # m_evaluation = m_train
        # evaluation_tuples = validation_tuples
        m_evaluation = self.m
        #evaluation_tuples = self.test_tuples  # test_tuples
        evaluation_tuples = [(x,y) for y in range(self.num_of_drugs) for x in range(y+1,self.num_of_drugs)]#self.test_tuples  # test_tuples
        evaluation_tuples = self.test_tuples
        print(f'evaluating {len(evaluation_tuples)} instances')
        eval_instances, _ = self.get_instances(evaluation_tuples,self.m)
        # preds_him = get_pred_him()
        print('done creating instances')
        preds = self.model_text.predict(eval_instances,
                                  batch_size=batch_size)  # [preds_him[x[0],x[1]] for x in evaluation_tuples] #

        if RANDOM_BASELINE:
            print("remove random")
            for i,val in enumerate(preds):
                rnd = random.random()
                if rnd >= RANDOM_POSITIVE_RATIO:
                    preds[i] = 1.0
                else:
                    preds[i] = 0.0
        print("remove ppp")
        # preds = self.model_text.predict(eval_instances,
        #                           batch_size=batch_size)  # [preds_him[x[0],x[1]] for x in evaluation_tuples] #

        print('done predicting', len(preds))
        count = 0
        predictions = np.zeros((self.m.shape[0], self.m.shape[1]))
        for i in range(len(evaluation_tuples)):
            idx1 = evaluation_tuples[i][0]
            idx2 = evaluation_tuples[i][1]
            predictions[idx1, idx2] = preds[i][0]
            count += 1
        eval_instances, _ = self.get_instances([(x[1], x[0]) for x in evaluation_tuples],self.m)

        preds = self.model_text.predict(eval_instances,
                                  batch_size=batch_size)  # [preds_him[x[1],x[0]] for x in evaluation_tuples]#
        print("after2 remove type(preds) type(eval_instances) ",type(preds),type(eval_instances))

        # preds = self.model_text.predict(eval_instances,
        #                           batch_size=batch_size)  # [preds_him[x[1],x[0]] for x in evaluation_tuples]#


        if RANDOM_BASELINE:
            print("remove random")
            for i,val in enumerate(preds):
                rnd = random.random()
                if rnd >= RANDOM_POSITIVE_RATIO:
                    preds[i] = 1.0
                else:
                    preds[i] = 0.0
        for i in range(len(evaluation_tuples)):
            idx1 = evaluation_tuples[i][0]
            idx2 = evaluation_tuples[i][1]
            predictions[idx2, idx1] = preds[i][0]  # must be reversed...
            count += 1
        for i in range(self.m.shape[0]):
            for j in range(i + 1, self.m.shape[0]):
                from scipy import stats
                # new_score = stats.hmean([ max(0.000001,predictions[i,j]),max(0.000001,predictions[j,i]) ])
                new_score = (predictions[i, j] + predictions[j, i]) / 2
                # new_score = max(predictions[i,j],predictions[j,i])
                # new_score = predictions[j,i]
                predictions[i, j] = new_score
                predictions[j, i] = new_score
        for i in range(self.m.shape[0]):
            assert predictions[i, i] == 0



        # predictions2 = np.zeros((self.m.shape[0], self.m.shape[1]))
        # G = nx.from_numpy_matrix(self.m)
        # for v1,v2 in  [(i, j) for i in G.nodes() for j in G.nodes() if j>i]:
        #     nn_preds1 = []
        #     for z in G[v2]:
        #         nn_preds1.append(predictions[v1, z])
        #     nn_preds2 = []
        #     for z in G[v1]:
        #         nn_preds2.append(predictions[v2, z])
        #     predictions2[v1, v2] = 0.5*predictions[v1, v2] + 0.5*np.mean(np.nan_to_num([np.mean(nn_preds1),np.mean(nn_preds2)]))
        #     predictions2[v2, v1] = predictions2[v1, v2]

        #predictions = predictions2

        self.predicted_mat = predictions
        print('predicted: ', count)
        s = set([(x[0],x[1])for x in self.test_tuples])
        predictions = [(i, v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(predictions)],
                                                   reverse=True)]  # get the cells of matrix in ascending order of cell value
        print(1)
        predictions = [(t, v) for t, v in predictions
                       if (t[0], t[1]) in s ]  # just half of the matrix and predictions larger than 0
        if self.predictions_pickle_file_name!=None:
            nn_predicted_mat = self.predicted_mat
            pickling_on = open(self.predictions_pickle_file_name, "wb")
            pickle.dump(nn_predicted_mat, pickling_on)
            pickling_on.close()

        return predictions

    def predict_combiner(self):

        # import math
        # import train_generator
        # importlib.reload(train_generator)
        # from train_generator import d2d_generators

        # d2d_generators_object = d2d_generators(m_train,train_tuples,validation_tuples)

        batch_size = 100000
        m_evaluation = self.m
        #evaluation_tuples = self.test_tuples  # test_tuples
        evaluation_tuples = self.test_tuples
        print(f'evaluating {len(evaluation_tuples)} instances')
        eval_instances, _ = self.get_instances(evaluation_tuples,self.m)
        # preds_him = get_pred_him()
        print('done creating instances')
        preds_text = self.model_text.predict(eval_instances,batch_size=batch_size)
        preds_emb = self.model_emb.predict(eval_instances,batch_size=batch_size)
        dat_x = np.array([preds_text[:,0], preds_emb[:,0]]).transpose()
        preds = self.combiner_model.predict(dat_x,batch_size=batch_size)

        if RANDOM_BASELINE:
            print("remove random")
            for i,val in enumerate(preds):
                rnd = random.random()
                if rnd >= RANDOM_POSITIVE_RATIO:
                    preds[i] = 1.0
                else:
                    preds[i] = 0.0
        print("remove ppp")
        # preds = self.model_text.predict(eval_instances,
        #                           batch_size=batch_size)  # [preds_him[x[0],x[1]] for x in evaluation_tuples] #

        print('done predicting', len(preds))
        count = 0
        predictions = np.zeros((self.m.shape[0], self.m.shape[1]))
        for i in range(len(evaluation_tuples)):
            idx1 = evaluation_tuples[i][0]
            idx2 = evaluation_tuples[i][1]
            predictions[idx1, idx2] = preds[i][0]
            count += 1
        eval_instances, _ = self.get_instances([(x[1], x[0]) for x in evaluation_tuples],self.m)

        preds_text = self.model_text.predict(eval_instances,batch_size=batch_size)
        preds_emb = self.model_emb.predict(eval_instances,batch_size=batch_size)
        dat_x = np.array([preds_text[:,0], preds_emb[:,0]]).transpose()
        preds = self.combiner_model.predict(dat_x,batch_size=batch_size)

        print("after2 remove type(preds) type(eval_instances) ",type(preds),type(eval_instances))

        # preds = self.model_text.predict(eval_instances,
        #                           batch_size=batch_size)  # [preds_him[x[1],x[0]] for x in evaluation_tuples]#


        if RANDOM_BASELINE:
            print("remove random")
            for i,val in enumerate(preds):
                rnd = random.random()
                if rnd >= RANDOM_POSITIVE_RATIO:
                    preds[i] = 1.0
                else:
                    preds[i] = 0.0
        for i in range(len(evaluation_tuples)):
            idx1 = evaluation_tuples[i][0]
            idx2 = evaluation_tuples[i][1]
            predictions[idx2, idx1] = preds[i][0]  # must be reversed...
            count += 1
        for i in range(self.m.shape[0]):
            for j in range(i + 1, self.m.shape[0]):
                from scipy import stats
                # new_score = stats.hmean([ max(0.000001,predictions[i,j]),max(0.000001,predictions[j,i]) ])
                new_score = (predictions[i, j] + predictions[j, i]) / 2
                # new_score = max(predictions[i,j],predictions[j,i])
                # new_score = predictions[j,i]
                predictions[i, j] = new_score
                predictions[j, i] = new_score
        for i in range(self.m.shape[0]):
            assert predictions[i, i] == 0



        # predictions2 = np.zeros((self.m.shape[0], self.m.shape[1]))
        # G = nx.from_numpy_matrix(self.m)
        # for v1,v2 in  [(i, j) for i in G.nodes() for j in G.nodes() if j>i]:
        #     nn_preds1 = []
        #     for z in G[v2]:
        #         nn_preds1.append(predictions[v1, z])
        #     nn_preds2 = []
        #     for z in G[v1]:
        #         nn_preds2.append(predictions[v2, z])
        #     predictions2[v1, v2] = 0.5*predictions[v1, v2] + 0.5*np.mean(np.nan_to_num([np.mean(nn_preds1),np.mean(nn_preds2)]))
        #     predictions2[v2, v1] = predictions2[v1, v2]

        #predictions = predictions2

        self.predicted_mat = predictions
        print('predicted: ', count)
        s = set([(x[0],x[1])for x in self.test_tuples])
        predictions = [(i, v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(predictions)],
                                                   reverse=True)]  # get the cells of matrix in ascending order of cell value
        print(1)
        predictions = [(t, v) for t, v in predictions
                       if (t[0], t[1]) in s ]  # just half of the matrix and predictions larger than 0
        if self.predictions_pickle_file_name!=None:
            nn_predicted_mat = self.predicted_mat
            pickling_on = open(self.predictions_pickle_file_name, "wb")
            pickle.dump(nn_predicted_mat, pickling_on)
            pickling_on.close()

        return predictions



def flatten_list(l):
    # return [item for sublist in l for item in sublist]
    return list(itertools.chain.from_iterable(l))

def create_d2d_sparse_matrix_rare(i2d,drug_to_interactions_train,drug_to_interactions_test):
    d2i = array_to_dict(i2d)
    number_of_drugs = len(d2i)
    print('creating matrix rare')
    column_names = ["drug_id", "interactions_train", "interactions_test"]
    rare_df = pd.DataFrame(columns=column_names)

    num_rare_drugs = 0
    rare_drugs = set()
    for x in sorted(drug_to_interactions_test.items()):
        if len(drug_to_interactions_train[x[0]]) < RARE_MAX:
            x2 = len(drug_to_interactions_train[x[0]])
            x3 = len(drug_to_interactions_test[x[0]])
            if (x3>x2):
                num_rare_drugs = num_rare_drugs+1
                rare_drugs.add(d2i[x[0]])
                rare_dict = {'drug_id':d2i[x[0]],'interactions_train':len(drug_to_interactions_train[x[0]]),'interactions_test':len(drug_to_interactions_test[x[0]])}
                rare_df = rare_df.append(rare_dict,ignore_index=True)
    print('num or rare drugs ',num_rare_drugs)
    rare_df.to_csv('rare_df.csv',sep=',')
    #     print('remove 0 ',x[0],d2i[x[0]])
    #     print('remove ',d2i[x[0]],len(drug_to_interactions_train[x[0]]))
    rows = flatten_list([[d2i[x[0]]] * len(x[1]) for x in sorted(drug_to_interactions_test.items()) if (len(drug_to_interactions_train[x[0]]) < RARE_MAX)])
    cols = [d2i[t] for t in flatten_list([x[1] for x in sorted(drug_to_interactions_test.items()) if len(drug_to_interactions_train[x[0]])< RARE_MAX])]
    #cols = [d2i[t] for t in flatten_list([x[1] for x in sorted(drug_to_interactions_test.items())])]
    print('number of rare interactions:',len(cols),len(rows),num_rare_drugs,number_of_drugs)
    assert len(rows) == len(cols)
    print('after assert')
    data = [1] * len(cols)
    #m = csr_matrix((data,(rows,cols)), shape=(number_of_drugs, number_of_drugs),dtype='f')
    m = csr_matrix((data, (rows, cols)), shape=(number_of_drugs, number_of_drugs), dtype='f')
    print('m shape:', m.shape, 'm non zeros:', m.nnz)
    m = m.todense()
    count_non_sym=0
    for i in range(m.shape[0]):
        for j in range(i+1,m.shape[0]):
            if m[i,j]!=m[j,i]:
                count_non_sym+=1
            m[i,j]=max(m[i,j],m[j,i])
            m[j, i] = m[i, j]
    print('non sym count (matrix was made sym using max):',count_non_sym)
    assert np.allclose(m, m.T, atol=1e-8) #matrix is symmetric
    return m,rare_drugs


def create_d2d_sparse_matrix(i2d, drug_to_interactions):
    d2i = array_to_dict(i2d)
    number_of_drugs = len(d2i)
    print('creating matrix')
    rows = flatten_list([[d2i[x[0]]] * len(x[1]) for x in sorted(drug_to_interactions.items())])
    cols = [d2i[t] for t in flatten_list([x[1] for x in sorted(drug_to_interactions.items())])]
    print('number of valid interactions:',len(cols))
    assert len(rows) == len(cols)
    data = [1] * len(cols)
    m = csr_matrix((data,(rows,cols)), shape=(number_of_drugs, number_of_drugs),dtype='f')
    print('m shape:', m.shape, 'm non zeros:', m.nnz)
    m = m.todense()
    count_non_sym=0
    for i in range(m.shape[0]):
        for j in range(i+1,m.shape[0]):
            if m[i,j]!=m[j,i]:
                count_non_sym+=1
            m[i,j]=max(m[i,j],m[j,i])
            m[j, i] = m[i, j]
    print('non sym count (matrix was made sym using max):',count_non_sym)
    assert np.allclose(m, m.T, atol=1e-8) #matrix is symmetric
    return m

def validate_intersections(i2d, interactions):
    for d in i2d:
        assert d in interactions
        assert len(interactions[d]) > 0 and d not in interactions[d]

def save_drug_names_to_file(drug_reader_obj, version):
        fname = os.path.join('output', 'data', 'drug_names' + version + '.csv')
        with open(fname, 'w') as f:
            for key in drug_reader_obj.keys():
                f.write("%s,%s\n" % (key, drug_reader_obj[key]))

def average_precision_at_k(k, class_correct):
    #return average precision at k.
    #more examples: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    #and: https://www.kaggle.com/c/avito-prohibited-content#evaluation
    #class_correct is a list with the binary correct label ordered by confidence level.
    assert k <= len(class_correct)
    assert k > 0
    score = 0.0
    hits = 0.0
    for i in range(k):
        if class_correct[i]==1:
            hits += 1.0
        #score += hits /(i+1.0)
    #score /= k
    score = hits/(1.0*k)
    return score

def create_train_test_split_relese(old_relese,new_relese):
    print('reading first file')
    d2d_releases_old = d2d_releases_reader()

    drug_reader_old, drug_preproc_old = d2d_releases_old.read_and_preproc_release(old_relese)
    if SAVE_DRUG_NAMES:
         save_drug_names_to_file(drug_reader_old.drug_id_to_name, old_relese)
    print('num interactions in old version:' ,sum([len(drug_preproc_old.valid_drug_to_interactions[x]) for x in drug_preproc_old.valid_drug_to_interactions])/2)
    print('num drugs old', len(drug_preproc_old.valid_drug_to_interactions))

    validate_intersections(drug_preproc_old.valid_drugs_array, drug_preproc_old.valid_drug_to_interactions)
    print('reading seconds file')
    d2d_releases_new = d2d_releases_reader()
    drug_reader_new, drug_preproc_new = d2d_releases_new.read_and_preproc_release(new_relese)
    print('num drugs new', len(drug_preproc_new.valid_drug_to_interactions))
    print('num interactions in new version:' ,sum([len(drug_preproc_new.valid_drug_to_interactions[x]) for x in drug_preproc_new.valid_drug_to_interactions])/2)

    validate_intersections(drug_preproc_new.valid_drugs_array, drug_preproc_new.valid_drug_to_interactions)
    print('preprocessing two versions')
    # interscting_i2d = drug_preproc_old.get_interscting_i2d(drug_preproc_new)
    interactions_older, interactions_newer, interscting_i2d = drug_preproc_old.get_intersecting_intersections(
        drug_preproc_new,ONLY_TEXT_VALID,MIN_COUNT)

    # interscting_i2d = sorted(list(set(drug_preproc_old.valid_drugs_array) & set(drug_preproc_new.valid_drugs_array)))
    # interactions_older, interactions_newer = drug_preproc_old.valid_drug_to_interactions, drug_preproc_new.valid_drug_to_interactions
    #print('intersecting drugs:', interscting_i2d)
    print('intersecting drugs len: ', len(interscting_i2d))

    # validate_intersections(interscting_i2d, interactions_older)
    # validate_intersections(interscting_i2d, interactions_newer)
    print('creating train matrix')
    m_train = create_d2d_sparse_matrix(interscting_i2d, interactions_older)
    print('creating test matrix')
    m_test = create_d2d_sparse_matrix(interscting_i2d, interactions_newer)
    m_test_rare,rare_drugs = create_d2d_sparse_matrix_rare(interscting_i2d,interactions_older,interactions_newer)
    print('remove after test ')
    evaluator = drug_evaluator(interscting_i2d, interactions_newer, interactions_older)
    test_tuples = drug_evaluator.get_nnz_tuples_from_marix(m_train, True)
    test_tuples_rare = drug_evaluator.get_nnz_tuples_from_marix_rare(m_train,True,rare_drugs)
    evaluation_type = 'release'
    # assert min(sum(np.asarray(m_train)))>0
    # assert min(sum(np.asarray(m_train.T)))>0
    # assert min(sum(np.asarray(m_test)))>0
    # assert min(sum(np.asarray(m_test.T)))>0



    return m_test, m_train, evaluator, test_tuples, interscting_i2d,evaluation_type,drug_reader_old.drug_id_to_name,rare_drugs,test_tuples_rare

def get_emb_file_path(version):
    textf = ""
    if ADD_TEXT:
        textf = "_text_"
    return os.path.join('output', 'data', 'embeddings_' + 'AMFP' + '_' + textf + version + '.csv')




def interaction_prediction():

    outputs = []
    overall_precision = []
    overall_k_precision = []
    overall_auc = []
    overall_ap = []
    overall_aupr = []
    overall_fpr_tpr = []
    overall_k_recall = []
    k_for_per_drug_precision = 50
    k_for_overall_precision = 100
    predictors = []

    def add_predictor_evaluation(preds, name):
        precision, recall, class_correct = evaluator.get_precision_recall(m_test, preds, test_tuples)
        class_correct_natural_sort = [x for _, x in sorted(zip(preds, [int(x * 1) for x in class_correct]))]
        preds_natural_sort = sorted(preds)
        outputs.append((class_correct_natural_sort, [x[1] for x in preds_natural_sort], name))

        precision_per_drug,recall_per_drug = evaluator.get_precision_per_drug(m_test, preds, k=k_for_per_drug_precision)
        overall_precision.append((precision, name))
        overall_k_precision.append((precision_per_drug, name))
        overall_k_recall.append((recall_per_drug, name))
        # AUC
        pr = [x[1] for x in preds]
        fpr, tpr, _ = roc_curve(class_correct, pr)
        auc = roc_auc_score(class_correct, pr)
        overall_auc.append((auc, name))
        average_precision = average_precision_at_k(k_for_overall_precision, class_correct)
        overall_ap.append((average_precision, name))
        overall_aupr.append(average_precision_score(class_correct, pr))
        overall_fpr_tpr.append((fpr, tpr, name))

    def add_predictor_evaluation_rare(preds, name,rare_drugs,test_tuples_rare):

        precision, recall, class_correct = evaluator.get_precision_recall_rare(m_test, preds, test_tuples_rare,rare_drugs)
        print('remove add_predictor_evaluation_rare predictions class_correct size '+str(len(preds))+' '+str(len(class_correct)))
        class_correct_natural_sort = [x for _, x in sorted(zip(preds, [int(x * 1) for x in class_correct]))]
        preds_natural_sort = sorted(preds)
        outputs.append((class_correct_natural_sort, [x[1] for x in preds_natural_sort], name))
        print('remove before precision per drug ')
        precision_per_drug,recall_per_drug = evaluator.get_precision_per_drug_rare(m_test, preds, test_tuples_rare,rare_drugs,k=k_for_per_drug_precision)
        print('remove after precision per drug ')
        overall_precision.append((precision, name))
        overall_k_precision.append((precision_per_drug, name))
        overall_k_recall.append((recall_per_drug,name))
        print('recall per drug ',recall_per_drug)
        # AUC
        preds_cp = preds.copy()
        test_tuples_rare_set = set(test_tuples_rare)
        pr = []
        for i,tuple in enumerate(preds_cp):
            drug_ids = tuple[0]
            if (drug_ids[0],drug_ids[1]) not in test_tuples_rare_set:
                continue
            pr.append(tuple[1])
        #pr = [x[1] for x in preds_cp]
        fpr, tpr, _ = roc_curve(class_correct, pr)
        auc = roc_auc_score(class_correct, pr)
        overall_auc.append((auc, name))
        average_precision = average_precision_at_k(k_for_overall_precision, class_correct)
        overall_ap.append((average_precision, name))
        overall_aupr.append(average_precision_score(class_correct, pr))
        overall_fpr_tpr.append((fpr, tpr, name))

    def get_interaction(row,m_test):
        val = m_test[(row['Tuple'])[0],(row['Tuple'])[1]]
        return val

    def tuple_to_string(my_list):
        val = " ".join(map(str, my_list['Tuple']))
        return val

    def compare_predictions(preds,preds2,name):
        precision, recall, class_correct = evaluator.get_precision_recall(m_test, preds, test_tuples)
        precision2, recall2, class_correct2 = evaluator.get_precision_recall(m_test, preds, test_tuples)

        pr1 = [x[1] for x in preds]
        un1 = [x[0] for x in preds]   #x[0] is the tuple of drugs, x[1] is the prediction, test_tuples is the set of test tuples
        pr2 = [x[1] for x in preds2]
        un2 = [x[0] for x in preds2]
        allList = {'pr1':pr1,'pr2:':pr2,'correct1':class_correct,'correct2':class_correct2,'unk1':un1,'unk2':un2}
        all_res = pd.DataFrame(allList)
        all_res.to_csv('F:\\Drugs\\Code\\all_res.csv')
        set_test = set(test_tuples)
        set_reg = set(un1)
        set_text = set(un2)
        diff_reg = set_test.difference(set_reg)
        diff_text = set_test.difference(set_text)
        df1 = pd.DataFrame(preds, columns=['Tuple','Pred'])
        df2 = pd.DataFrame(preds2, columns=['Tuple','Pred2'])
        df1['original'] = df1.apply(lambda row: get_interaction(row,m_test), axis=1)
        df1.to_csv('F:\\Drugs\\Code\\df1_2.csv')
        #df1['Tuple'] = df1['Tuple'].astype(str)
        #df2['Tuple'] = df2['Tuple'].astype
        df1['Tuple2'] = df1.apply(lambda row:tuple_to_string(row),axis=1)
        df2['Tuple2'] = df2.apply(lambda row:tuple_to_string(row),axis=1)
        #df1.join(df2,on='Tuple',how='inner', rsuffix='_text')
        df1 = df1.merge(df2,left_on='Tuple2',right_on='Tuple2',suffixes=('_left', '_right'))
        #df1.merge(df2,on='Tuple2')
        df1.to_csv('F:\\Drugs\\Code\\df1.csv')

        #mtest[df1['Tuple'][0],df1['Tuple'][1]]>0

    def save_data(m_train,m_test,test_tuples,predictions):
        print("m_train ",type(m_train))
        print("m_test ", type(m_test))
        print("test_tuples ", type(test_tuples))
        print("predictions ", type(predictions))

    def save_dict(dict,in_fname):
        fname = os.path.join('output', in_fname)
        print('remove save_dict '+fname)
        try:
            w = csv.writer(open(fname, "w"))
            print('remove save_dict 2'+'after open')
            for key, val in dict.items():
                w.writerow([key, val])
        except:
            print('save_dict error writing')

    def save_list(my_list,in_fname):
        fname = os.path.join('output', in_fname)
        try:
            with open(fname, 'w') as f:
                for count,item in enumerate(my_list):
                    f.write(str(count)+','+str(item)+'\n')
        except:
            print('save list error writing '+fname+','+str(item))

    #for version_details in d2d_versions_metadata:
    try:

        ## Init
        ## This part should be based on parameters file
        #version = version_details['VERSION']
        # version_details = "5.1.6"
        # version = "5.1.6"
        # new_version = version
        # old_version = "5.1.5"
        version_details = "5.1.1"
        version = "5.1.1"
        new_version = version
        old_version = "5.0.0"
        textf = ""
        if ADD_TEXT:
             textf = "_otext_"

        evaluation_method = "Retrospective"
        nn_p_file = os.path.join('output', 'predictions_'+
                                     evaluation_method + old_version + new_version + textf + "nn_predictions.p")
        nn_m_file = os.path.join('pickles', 'model_' +
                                     evaluation_method + old_version + new_version + textf + "nn")

        #amfp_params = {'mul_emb_size': 512, 'dropout': 0.4, 'epochs': 10, 'batch_size': 1024, 'learning_rate': 0.01,
        #               'propagation_factor': 0.4}
        amfp_params = {'mul_emb_size': 512, 'dropout': 0.4, 'epochs': 10, 'batch_size': 1024, 'learning_rate': 0.01,
                         'propagation_factor': 0.4, 'text_propagation_factor':0.5, 'text_propagation_factor2':0.0}

        print("chaneed remove ")
        validaition_instance_features = None
        target_att = None
        #old_version

        m_test, m_train, evaluator, test_tuples, i2d, evaluation_type, drug_id_to_name,rare_drugs,test_tuples_rare = create_train_test_split_relese(
            old_relese=old_version, new_relese=new_version)

        if ADD_TEXT:
            nn = drugs_text_nn_predictor(m_train, test_tuples, validation_tuples=validaition_instance_features,
                                validation_target=target_att, name='AMFP', **amfp_params)
        else:
            nn = drugs_nn_predictor(m_train, test_tuples, validation_tuples=validaition_instance_features,
                                validation_target=target_att, name='AMFP', **amfp_params)

        if ADD_TEXT:
            nn.set_drug_id_to_name(drug_id_to_name,i2d)
            nn.set_concept_embedding_matrix()

        #print("drug id to name ex ",type(drug_id_to_name[0]))

        if MODEL_TRAIN:
            nn.fit()
        else:
            nn.model_file_name = nn_m_file
            nn.load_model()


        if MODEL_TRAIN:
            embeddings = [np.concatenate([[i2d[i]], x]) for i, x in enumerate(nn.get_embeddings())]
            emb = pd.DataFrame(embeddings)
            emb.columns = [drugBank_id] + ['emb_mult_dim_' + str(x) for x in range(amfp_params['mul_emb_size'])] + [
                'emb_add_dim_0']
            emb = emb.set_index(drugBank_id)
            emb = emb.astype(float)
            nn.predictions_pickle_file_name = nn_p_file
            nn.model_file_name = nn_m_file

            if USE_DB:
                db_engine = get_DB_connection()
                con = db_engine.connect()
                emb.to_sql(combine_table_name_version(AMFP_table, old_version), con=con, schema=AMFP_schema, index=True,
                       if_exists='replace')
                con.close()
                db_engine.dispose()
            else:
                fname = get_emb_file_path(old_version)
                ##emb.to_csv(fname)
                ##print("remove before sav3 ")
                ##nn.save_model()
                ##print("remove after save 4")
        predictions = nn.predict()
        try:
            name = ""
            if ADD_TEXT:
                name = 'text_baseline'
            else:
                name = 'guy_baseline'
            add_predictor_evaluation(preds=predictions, name=name)
            add_predictor_evaluation_rare(preds=predictions, name=name+' rare',test_tuples_rare=test_tuples_rare,rare_drugs=rare_drugs)

        except:
            print('problem adding evaluation')



        # save_data(m_train,m_test,test_tuples,predictions)
        # for i in range(len(overall_auc)):
        #     print(
        #         f'Name: {overall_auc[i][1]}, auc: {overall_auc[i][0]}, map: {overall_ap[i][0]}, aupr: {overall_aupr[i]}, AP@k: {overall_k_precision[i][0][0]},{overall_k_precision[i][0][1]},{overall_k_precision[i][0][2]},{overall_k_precision[i][0][3]},{overall_k_precision[i][0][4]}')

        if ADD_TEXT:
            predictions_text = nn.predict_text()
            add_predictor_evaluation(preds=predictions_text,name='only text')
            add_predictor_evaluation_rare(preds=predictions_text, name='only text rare',test_tuples_rare=test_tuples_rare,rare_drugs=rare_drugs)
        # for i in range(len(overall_auc)):
        #     print(
        #         f'Name: {overall_auc[i][1]}, auc: {overall_auc[i][0]}, map: {overall_ap[i][0]}, aupr: {overall_aupr[i]}, AP@k: {overall_k_precision[i][0][0]},{overall_k_precision[i][0][1]},{overall_k_precision[i][0][2]},{overall_k_precision[i][0][3]},{overall_k_precision[i][0][4]}')
         #   in_model = keras.models.load_model(nn.checkpoint_filepath2)
            # nn.set_model_text(keras.models.load_model(nn.checkpoint_filepath2))
            # predictions_text = nn.predict_text()
            #
            # add_predictor_evaluation(preds=predictions_text, name='only text original')
            # add_predictor_evaluation_rare(preds=predictions_text, name='only text original rare',test_tuples_rare=test_tuples_rare,rare_drugs=rare_drugs)
            predictions_combiner = nn.predict_combiner()
            add_predictor_evaluation(preds=predictions_combiner, name='combiner')
            add_predictor_evaluation_rare(preds=predictions_combiner, name='combiner rare',test_tuples_rare=test_tuples_rare,rare_drugs=rare_drugs)


        for i in range(len(overall_auc)):
            #print(
            #    f'Name: {overall_auc[i][1]}, auc: {overall_auc[i][0]}, map: {overall_ap[i][0]}, aupr: {overall_aupr[i]}, AP@k: {overall_k_precision[i][0][0]},{overall_k_precision[i][0][1]},{overall_k_precision[i][0][2]},{overall_k_precision[i][0][3]},{overall_k_precision[i][0][4]}')
            print(
                f'Name: {overall_auc[i][1]}, auc: {overall_auc[i][0]}, map: {overall_ap[i][0]}, aupr: {overall_aupr[i]}, AP@k: {overall_k_precision[i][0][0]},{overall_k_precision[i][0][1]},{overall_k_precision[i][0][2]},{overall_k_precision[i][0][3]},{overall_k_precision[i][0][4]}')
            save_list(overall_k_precision[i][0],'ap_k_drug3'+str(overall_auc[i][1])+'.csv')
            save_list(overall_k_recall[i][0],'rec_k_drug3'+str(overall_auc[i][1])+'.csv')


        #compare_predictions(predictions_text,predictions,'compare')


    except:
        print('#########################################cannot read x',version_details)


interaction_prediction()
