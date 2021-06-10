import os
import pandas as pd
import keras
from keras import layers, regularizers
from keras.layers import multiply, Multiply, Concatenate
from pandas import HDFStore
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def getPCA(X_unlabled,features,modality,k=100):
    # import prince
    # pca = prince.MCA(
    #     n_components=1000,
    #  #n_iter=3,
    #  copy=True,
    #  check_input=True,
    #  engine='auto',
    #  random_state=42 )
    # X_unlabled_pca = pca.fit_transform(
    #     VarianceThreshold().fit_transform(
    #         X_unlabled[modalities.loc[modalities.modality.isin([modality]), 'feature']]
    #     )
    # )

    #pca = corex.Corex(n_hidden=50, dim_hidden=10, marginal_description='discrete', smooth_marginals=False,max_iter=100,n_repeat=1,n_cpu=None,verbose=True) # n_hidden = dim of output. dim_hiden=card of output#too slow
    # Define the number of hidden factors to use (n_hidden=2).
    # And each latent factor is binary (dim_hidden=2)
    # marginal_description can be 'discrete' or 'gaussian' if your data is continuous
    # smooth_marginals = True turns on Bayesian smoothing
    #layer1.fit(X)  # Fit on data.
    k = round(len(features) ** (0.5) )
    pca = PCA(n_components=k)
    X_unlabled_pca = pca.fit_transform(
       StandardScaler().fit_transform(X_unlabled[features]))

    X_unlabled_pca = pd.DataFrame(X_unlabled_pca)
    X_unlabled_pca.index = X_unlabled.index
    X_unlabled_pca.columns = [modality+'_PCA' +"_" + str(x) for x in range(pca.n_components)]
    return X_unlabled_pca

def getselectK(X_unlabled, modality,t=0.01):
    select = VarianceThreshold(t)
    data = X_unlabled[modalities.loc[modalities.modality.isin([modality]), 'feature']]
    select.fit(data)
    data = data.iloc[:,select.get_support()]
    data.columns = [x+ ' select' for x in data.columns]
    return data



def extract_text_features(X_unlabled, modalities):
    count_vect = CountVectorizer(binary=False)
    X_unlabled.text = X_unlabled.text.fillna('')
    word_counts = count_vect.fit_transform(
        X_unlabled[modalities.loc[modalities.modality.isin(['text']), 'feature']].text)
    text_features = ['Mention: ' + x for x in count_vect.get_feature_names()]
    text_features = pd.DataFrame(word_counts.toarray(), columns=text_features, index=X_unlabled.index)
    X_unlabled = X_unlabled.join(text_features, how='left')
    for c in text_features:
        modalities = modalities.append({'modality': 'text_processed', 'feature': str(c)}, ignore_index=True)
    return X_unlabled, modalities

def get_col_clusters(X_unlabled,modalities):
    cat_data = X_unlabled[modalities.loc[modalities.modality.isin(['Category','ATC_Level_2_description','ATC_Level_3_description','ATC_Level_4_description']), 'feature']]
    cat_data = cat_data.drop('Number of Category', axis=1)
    s = cat_data.columns.to_series().str.replace('Category: ', "")
    s = s.str.replace('ATC Level 2 description: ', "")
    s = s.str.replace('ATC Level 3 description: ', "")
    s = s.str.replace('ATC Level 4 description: ', "")

    count_vect = CountVectorizer(binary=False)
    word_counts = count_vect.fit_transform(s)
    tf = TfidfTransformer()
    word_counts = tf.fit_transform(word_counts)
    text_features = count_vect.get_feature_names()
    text_features = pd.DataFrame(word_counts.toarray(), columns=text_features, index=s.index)
    ans = None
    dist_mat = pairwise_distances(text_features.values, metric='jaccard', n_jobs=4)
    for n in [2500]:
        kmeans = AgglomerativeClustering(n_clusters=n,linkage='average',affinity='precomputed')#affinity=sklearn.metrics.jaccard_score ,affinity='cosine'
        #kmeans.fit(text_features)
        clusters = pd.DataFrame(kmeans.fit_predict(dist_mat),columns=['cluster'],index=text_features.index)
        for g in clusters.groupby('cluster').groups:
            g= clusters.groupby('cluster').groups[g]
            g_col = X_unlabled[g].sum(axis=1)
            if ans is None:
                ans = pd.DataFrame(g_col)
            else:
                ans[str(g[0])+"_c_"+str(n)] = g_col
        print('done clustering',n)

    print('done clustering')
    return ans

def convert_to_one_hot(X_unlabled, modalities, modality):
    cols = modalities.loc[modalities.modality.isin([modality]), 'feature']
    X_unlabled =  pd.get_dummies(X_unlabled,columns=cols,prefix=cols,prefix_sep=': ')
    modalities=modalities[modalities.modality!=modality]
    for c in X_unlabled.columns:
        if c.split(': ')[0] in cols.values:
            modalities = modalities.append({'modality': modality, 'feature': c}, ignore_index=True)
    return X_unlabled,modalities


def supervised_dim_reduction(X_train, y_train, X_test=None, validation=0.0):
    if X_test is None:
        X_test =X_train
    inputs = keras.Input(shape=(len(X_train.columns),))
    reg = None#regularizers.l1_l2(l1=1e-5, l2=1e-5)
    dense_out = layers.Dense(len(y_train.columns), activation="sigmoid", kernel_regularizer=reg)#for some weird reason softmax works better

    # mult = Multiply()([inputs, inputs])
    # layer = Concatenate()([mult,inputs])
    layer= inputs
    layers_list = []
    for i in range(10):
        layer = layers.Dropout(0.0)(layer)
        layer = layers.Dense(300, activation="relu",kernel_regularizer=reg)(layer)
        layers_list.append(layer)

    outputs = dense_out(layer)
    model = keras.Model(inputs=inputs, outputs=outputs, name="dim_reduction")
    model.compile(
        loss='binary_crossentropy',
        optimizer='Adam',
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train.astype(int), shuffle=True, validation_split=validation, epochs=25, verbose=2)#epochs=25

    model_out = keras.Model(inputs=inputs, outputs=layers_list[0], name="dim_reduction")
    model_out.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    ans = model_out.predict(X_test)
    ans = pd.DataFrame(ans)
    return ans#,model.predict(X_test)

if __name__ =='__main__':
    os.chdir('..\\..')
    store = HDFStore('output\data\modalities_dict.h5')
    X_unlabled = store['df']
    modalities = store['modalities']
    cols_x = modalities[modalities.modality.isin(['mol2vec'])].feature
    cols_y = modalities[modalities.modality.isin(['Category'])].feature
    X = X_unlabled[cols_x]
    y = X_unlabled[cols_y].drop('Number of Category', axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.33, random_state=42)
    ans = supervised_dim_reduction(X,y,validation=0.2)
    # 1 - y.sum().sum() / y.count().sum()

    # print(np.sum(preds.round() == y_test, axis=0).sum() / y_test.count().sum())