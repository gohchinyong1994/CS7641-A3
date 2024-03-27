# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:18:43 2024

@author: User
"""

import numpy as np 
import pandas as pd
import mlrose_hiive
import time
import matplotlib.pyplot as plt
import dataframe_image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer, classification_report, silhouette_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
import timeit
from functools import partial
import time

from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import random_projection
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.cluster import KElbowVisualizer

import umap
import umap.plot

os.chdir(r"D:\Georgia Tech\CS7641 Machine Learning\A3")

from A3_utils import getBananaData, getDiabetesData, generateClusterResults, generatePCAResults, generateICAResults, generateRPResults, generateUMAPResults, plotProjection

pd.set_option('display.max_columns', None)


df1_split, df1_scaler = getBananaData()
df2_split, df2_scaler = getDiabetesData()

generateClusterResults(*df1_split, alg='EM'    , dLabel='Banana')
generateClusterResults(*df1_split, alg='KMeans', dLabel='Banana')
generateClusterResults(*df2_split, alg='EM'    , dLabel='Diabetes')
generateClusterResults(*df2_split, alg='KMeans', dLabel='Diabetes')

generatePCAResults(df1_split, n=df1_split[0].shape[1], dLabel='Banana') #n=7 - use 4
generatePCAResults(df2_split, n=df2_split[0].shape[1], dLabel='Diabetes') #n=8 - use 4

generateICAResults(df1_split, n=11, dLabel='Banana') # n=7 - use 4
generateICAResults(df2_split, n=11, dLabel='Diabetes') # n=6 - use 4

generateRPResults(df1_split, n=13, dLabel='Banana') # use 4
generateRPResults(df2_split, n=13, dLabel='Diabetes') # use 4

df1_nn_list = [5,25] + [int(df1_split[0].shape[0]/x) for x in range(2,0,-1)]
generateUMAPResults(df1_split, df1_nn_list, dLabel='Banana') # use 4
df2_nn_list = [5,25] + [int(df2_split[0].shape[0]/x) for x in range(2,0,-1)]
generateUMAPResults(df2_split, df2_nn_list, dLabel='Diabetes') # use 4

# Part 3
pca1 = PCA(n_components=4, random_state=1337).fit(df1_split[0])
df1_split_pca = [pca1.transform(df1_split[0]), pca1.transform(df1_split[1]), 
                 df1_split[2].copy(), df1_split[3].copy()]

pca2 = PCA(n_components=5, random_state=1337).fit(df2_split[0])
df2_split_pca = [pca2.transform(df2_split[0]), pca2.transform(df2_split[1]), 
                 df2_split[2].copy(), df2_split[3].copy()]

generateClusterResults(*df1_split_pca, alg='EM with PCA'    , dLabel='Banana')
generateClusterResults(*df1_split_pca, alg='KMeans with PCA', dLabel='Banana')
generateClusterResults(*df2_split_pca, alg='EM with PCA'    , dLabel='Diabetes')
generateClusterResults(*df2_split_pca, alg='KMeans with PCA', dLabel='Diabetes')

ica1 = FastICA(n_components=4, max_iter=500, random_state=1337).fit(df1_split[0])
df1_split_ica = [ica1.transform(df1_split[0]), ica1.transform(df1_split[1]), 
                 df1_split[2].copy(), df1_split[3].copy()]

ica2 = FastICA(n_components=4, max_iter=500, random_state=1337).fit(df2_split[0])
df2_split_ica = [ica2.transform(df2_split[0]), ica2.transform(df2_split[1]), 
                 df2_split[2].copy(), df2_split[3].copy()]


generateClusterResults(*df1_split_ica, alg='EM with ICA'    , dLabel='Banana')
generateClusterResults(*df1_split_ica, alg='KMeans with ICA', dLabel='Banana')
generateClusterResults(*df2_split_ica, alg='EM with ICA'    , dLabel='Diabetes')
generateClusterResults(*df2_split_ica, alg='KMeans with ICA', dLabel='Diabetes')

rp1 = random_projection.SparseRandomProjection(n_components=4, random_state=1337).fit(df1_split[0])
df1_split_rp = [rp1.transform(df1_split[0]), rp1.transform(df1_split[1]), 
                 df1_split[2].copy(), df1_split[3].copy()]

rp2 = random_projection.SparseRandomProjection(n_components=4, random_state=1337).fit(df2_split[0])
df2_split_rp = [rp2.transform(df2_split[0]), rp2.transform(df2_split[1]), 
                 df2_split[2].copy(), df2_split[3].copy()]

generateClusterResults(*df1_split_rp, alg='EM with RP'    , dLabel='Banana')
generateClusterResults(*df1_split_rp, alg='KMeans with RP', dLabel='Banana')
generateClusterResults(*df2_split_rp, alg='EM with RP'    , dLabel='Diabetes')
generateClusterResults(*df2_split_rp, alg='KMeans with RP', dLabel='Diabetes')


um1 = umap.UMAP(n_components=4,n_neighbors = 25).fit(df1_split[0])
df1_split_um = [um1.transform(df1_split[0]), um1.transform(df1_split[1]), 
                 df1_split[2].copy(), df1_split[3].copy()]

um2 = umap.UMAP(n_components=4,n_neighbors = 25).fit(df2_split[0])
df2_split_um = [um2.transform(df2_split[0]), um2.transform(df2_split[1]), 
                 df2_split[2].copy(), df2_split[3].copy()]

generateClusterResults(*df1_split_um, alg='EM with UMAP'    , dLabel='Banana')
generateClusterResults(*df1_split_um, alg='KMeans with UMAP', dLabel='Banana')
generateClusterResults(*df2_split_um, alg='EM with UMAP'    , dLabel='Diabetes')
generateClusterResults(*df2_split_um, alg='KMeans with UMAP', dLabel='Diabetes')

# Plot Projections
plotProjection( df1_split[0], alg='Base', dLabel='Banana' )
plotProjection( df2_split[0], alg='Base', dLabel='Diabetes' )

plotProjection( df1_split_pca[0], alg='PCA', dLabel='Banana' )
plotProjection( df2_split_pca[0], alg='PCA', dLabel='Diabetes' )

plotProjection( df1_split_ica[0], alg='ICA', dLabel='Banana' )
plotProjection( df2_split_ica[0], alg='ICA', dLabel='Diabetes' )

plotProjection( df1_split_rp[0], alg='RP', dLabel='Banana' )
plotProjection( df2_split_rp[0], alg='RP', dLabel='Diabetes' )

plotProjection( df1_split_um[0], alg='UMAP', dLabel='Banana' )
plotProjection( df2_split_um[0], alg='UMAP', dLabel='Diabetes' )

# Part 4
#x_train, x_test, y_train, y_test = df1_split
for split in [df1_split, df1_split_pca, df1_split_ica, df1_split_rp, df1_split_um]:
#x_train, x_test, y_train, y_test = df2_split
#for split in [df2_split, df2_split_pca, df2_split_ica, df2_split_rp, df2_split_um]:

    x_train, x_test, y_train, y_test = split
   
    rand = np.random.RandomState(seed=1337)
    nnc = MLPClassifier(activation='logistic', random_state=rand)
    strat_kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=rand)
    
    #nnc_params = {'hidden_layer_sizes': [15], 'learning_rate_init': [0.01]}
    hidden_layers_list = [1,5,10,15,20]
    lr_list = [0.001, 0.01, 0.1]
    nnc_params = {'learning_rate_init': lr_list, 'hidden_layer_sizes': hidden_layers_list, }
    start_time = time.time()
    grid_search_nnc = GridSearchCV(estimator=nnc, param_grid=nnc_params, scoring='f1', cv=strat_kfold, n_jobs=-1, return_train_score=True)
    grid_search_nnc.fit(x_train, y_train)
    print(time.time()-start_time)
    
    cv_score = grid_search_nnc.best_score_
    test_score = grid_search_nnc.score(x_test, y_test)
    print('Grid Search NNC')
    print(grid_search_nnc.best_params_)
    print('CV F1 Score: %.5f' % (cv_score*100))
    print('Test F1 Score: %.5f' % (test_score*100))

# Part 5
# Base Case
for split in [df1_split]:
    x_train, x_test, y_train, y_test = split
   
    rand = np.random.RandomState(seed=1337)
    nnc = MLPClassifier(activation='logistic', random_state=rand)
    strat_kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=rand)
    
    #nnc_params = {'hidden_layer_sizes': [15], 'learning_rate_init': [0.01]}
    hidden_layers_list = [1,5,10,15,20]
    lr_list = [0.001, 0.01, 0.1]
    nnc_params = {'learning_rate_init': lr_list, 'hidden_layer_sizes': hidden_layers_list, }
    start_time = time.time()
    grid_search_nnc = GridSearchCV(estimator=nnc, param_grid=nnc_params, scoring='f1', cv=strat_kfold, n_jobs=-1, return_train_score=True)
    grid_search_nnc.fit(x_train, y_train)
    print(time.time()-start_time)
    
    cv_score = grid_search_nnc.best_score_
    test_score = grid_search_nnc.score(x_test, y_test)
    print('Grid Search NNC')
    print(grid_search_nnc.best_params_)
    print('CV F1 Score: %.5f' % (cv_score*100))
    print('Test F1 Score: %.5f' % (test_score*100))


def oneHotArray(x, y): 
    #y containing the onehot labels
    cluster_cols=np.array(pd.get_dummies(pd.DataFrame(y.astype(str),columns=['cluster'])).astype('int'))
    return np.concatenate([x,cluster_cols],axis=1)

x_train, x_test, y_train, y_test = df1_split

for alg in ['EM', 'KMeans']:
    if alg == 'EM':
        clf = GaussianMixture(4, random_state=1337)
    else:
        clf = KMeans(4, random_state=1337)
    train_label = clf.fit_predict( x_train )
    test_label = clf.predict( x_test )
    
    x_train_combined = oneHotArray(x_train, train_label)
    x_test_combined = oneHotArray(x_test, test_label)
    
    rand = np.random.RandomState(seed=1337)
    nnc = MLPClassifier(activation='logistic', random_state=rand)
    strat_kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=rand)
    
    hidden_layers_list = [1,5,10,15,20]
    lr_list = [0.001, 0.01, 0.1]
    nnc_params = {'learning_rate_init': lr_list, 'hidden_layer_sizes': hidden_layers_list, }
    
    start_time = time.time()
    grid_search_nnc = GridSearchCV(estimator=nnc, param_grid=nnc_params, scoring='f1', cv=strat_kfold, n_jobs=-1, return_train_score=True)
    grid_search_nnc.fit(x_train_combined, y_train)
    print(time.time()-start_time)
    
    cv_score = grid_search_nnc.best_score_
    test_score = grid_search_nnc.score(x_test_combined, y_test)
    print('Grid Search NNC: %s' % alg)
    print(grid_search_nnc.best_params_)
    print('CV F1 Score: %.5f' % (cv_score*100))
    print('Test F1 Score: %.5f' % (test_score*100))












