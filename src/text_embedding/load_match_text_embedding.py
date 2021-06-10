import itertools
import os
#import tensorflow
import pickle
import random
import keras.backend as K
import networkx as nx
import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Add, Dropout, Multiply, Concatenate, Flatten, Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics import roc_auc_score
from src.data_readers.d2d_releases_reader import d2d_releases_reader
import pandas as pd
from gensim.models import KeyedVectors
import sys, json
from gensim.test.utils import datapath

from src.data_readers.dataset_dates import d2d_versions_metadata
from src.drug_classification.single_drug_features_creator import DrugBank_drug_feature_creator
from src.main.data_migration.table_names import get_DB_connection, AMFP_schema, combine_table_name_version, AMFP_table, \
    drugBank_id

os.chdir('..\\..')

def save_concept_names_to_file(concept_reader_obj):
    fname = os.path.join('output', 'data', 'concept_names' + '.csv')
    with open(fname, 'w',encoding='utf-8') as f:
        for key in concept_reader_obj.keys():
            f.write(key+"\n")

def read_concept_embedding_file():
    import json
    fname = 'F:\Drugs\Concept_embedding2020\concept_cbow.json'
    # read file
    with open(fname, 'r') as myfile:
        data = myfile.read()

    # parse file
    obj = json.loads(data)
    save_concept_names_to_file(obj)

def load_gensim_embedding(path, binary):
    print("path ",path)
    embedding = KeyedVectors.load_word2vec_format(datapath(path), binary = binary)
    print('embedding loaded from', path)
    return embedding

def read_concept_embedding_bin_file():
    YOUR_BIOCONCEPTVEC_PATH = 'F:\Drugs\Concept_embedding2020\\bioconceptvec_word2vec_cbow.bin'
    model = load_gensim_embedding(YOUR_BIOCONCEPTVEC_PATH, binary=True)
    return model

def add_text_embedding():
    return

def match_concept_embedding_with_drug_bank_entity():

    model = read_concept_embedding_bin_file()
    print("cwd ",os.getcwd())
    drugb_drugs_fname = os.path.join('output', 'data', 'drug_names5.1.5' + '.csv')
    #drugb_drugs_fname = "F:\\Drugs\\Code\\DDI_prediction_mm\\multimodal_learning-master\\output\\data\\drug_names5.1.5.csv"
    #drugb_concept_drugs_fname = "F:\\Drugs\\Code\\DDI_prediction_mm\\multimodal_learning-master\\output\\data\\drug_names5.1.5_con.csv"
    drugb_concept_drugs_fname = os.path.join('output', 'data', 'drug_names5.1.5_con' + '.csv')

    db_df = pd.read_csv(drugb_drugs_fname)
    db_df["concept_name"] = ""

    for index, row in db_df.iterrows():
        words = (row['name']).lower().split()
        if len(words) == 1:
            try:
                vec = model[words[0]]
                db_df.iloc[index, 3] = words[0]
            except:
                continue
        else:
            word_str = words[0]
            try:
                vec = model[word_str]
                db_df.iloc[index, 3] = word_str
            except:
                continue
            for i in range(1, len(words)):
                word_str = word_str + "-" + words[i]
                try:
                    vec = model[word_str]
                    db_df.iloc[index, 3] = word_str
                except:
                    continue

    db_df.to_csv(drugb_concept_drugs_fname)

if __name__ == "__main__":
    match_concept_embedding_with_drug_bank_entity()