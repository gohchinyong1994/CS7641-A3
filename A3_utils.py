# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:48:26 2024

@author: User
"""

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from imblearn.over_sampling import SMOTE
import time
import random
from sklearn.decomposition import PCA, FastICA
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn import random_projection
from sklearn.metrics import mean_squared_error
import umap

def getBananaData(seed=1337):
    # Source: https://www.kaggle.com/datasets/l3llff/banana
    df1 = pd.read_csv('banana_quality.csv')

    df1.loc[df1['Quality'] == 'Good', 'Quality'] = 1
    df1.loc[df1['Quality'] == 'Bad', 'Quality'] = 0
    df1['Quality'] = df1['Quality'].astype('int')
    
    x_var = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness',
           'Acidity']
    
    y_var = ['Quality']
    

    df1_split = train_test_split(np.array(df1.loc[:,x_var]),np.array(df1.loc[:,y_var]),
                                                        test_size=0.80,random_state=seed)
    
    scaler = StandardScaler()
    df1_split[0] = scaler.fit_transform(df1_split[0])
    df1_split[1] = scaler.transform(df1_split[1])
    return df1_split, scaler


'''
def getCreditCardData(seed=1337):
    # Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    df_cc = pd.read_csv('creditcard.csv')
    
    x_var = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    y_var = ['Class']

    # Oversampled and resampled for performance purposes
    oversample = SMOTE()
    x, y = oversample.fit_resample(np.array(df_cc.loc[:,x_var]), np.array(df_cc.loc[:,y_var]))
    x, y = zip(*random.sample(list(zip(x, y)), 1600))
    df2_split = train_test_split(np.array(x), np.array(y), test_size=0.80,random_state=seed)
    return df2_split
'''

def getDiabetesData(seed=1337):
    # Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    
    df = pd.read_csv('diabetes.csv')
    
    x_var = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    y_var = ['Outcome']
    
    # remove zero data
    df = df.loc[~(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction']]==0).any(axis=1)]
    # Oversampled since only 130/392 are positive
    oversample = SMOTE()
    x, y = oversample.fit_resample(np.array(df.loc[:,x_var]), np.array(df.loc[:,y_var]))
    df_split = train_test_split(np.array(x), np.array(y), test_size=0.10,random_state=seed)
    
    scaler = StandardScaler()
    df_split[0] = scaler.fit_transform(df_split[0])
    df_split[1] = scaler.transform(df_split[1])
    return df_split, scaler


def generateClusterResults(x_train, x_test, y_train, y_test, alg='EM', dLabel='Banana', seed=1337, verbose=False):
    start_time = time.time()
    
    list_clusters = list(np.arange(2,21))
    sil_scores = []
    cluster_data = [] # List containing N, Cluster, TrainOrTest, Representing, Acc
    train_accs = []
    test_accs = []
    
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()
    for n in list_clusters:
        if alg.startswith('EM'):
            clf = GaussianMixture(n, random_state=seed)
        elif alg.startswith('KMeans'):
            clf = KMeans(n, random_state=seed)
            #clf = KMeans(n)
        else:
            raise Exception('Model not implemented')
        label = clf.fit_predict( x_train )
        sil_score = silhouette_score( x_train, label )
        sil_scores.append(sil_score)
        correct_matches = 0
        label_map = {}
        for m in np.arange(n):
            cluster_sum = np.count_nonzero(np.where(label==m,1,0))
            cluster_1_matches = np.count_nonzero(np.where(label==m,1,-1) == y_train_flat)
            cluster_0_matches = np.count_nonzero(np.where(label==m,0,-1) == y_train_flat)
            if cluster_1_matches > cluster_0_matches:
                correct_matches += cluster_1_matches
                label_map[m] = 1
                cluster_acc = cluster_1_matches/cluster_sum if cluster_sum != 0 else 0.0
                cluster_data.append([n, m, 'Train', 1, cluster_acc])
                if verbose:
                    print('%d: Cluster %d represents 1 with train accuracy of %.3f' % (n, m, cluster_acc))
            else:
                correct_matches += cluster_0_matches
                label_map[m] = 0
                cluster_acc = cluster_0_matches/cluster_sum if cluster_sum != 0 else 0.0
                cluster_data.append([n, m, 'Train', 0, cluster_acc])
                if verbose:
                    print('%d: Cluster %d represents 0 with train accuracy of %.3f' % (n, m, cluster_acc))
        
        train_acc = correct_matches/len(y_train_flat)
        train_accs.append(train_acc)
        if verbose:
            print('%d: Train Accuracy: %.3f' % (n, train_acc))
        
        test_label = clf.predict(x_test)
        test_correct_matches = 0
        for m in np.arange(n):
            cluster_sum = np.count_nonzero(np.where(test_label==m,1,0))
            if label_map[m] == 1:
                cluster_1_matches = np.count_nonzero(np.where(test_label==m,1,-1) == y_test_flat)
                test_correct_matches += cluster_1_matches
                cluster_acc = cluster_1_matches/cluster_sum if cluster_sum != 0 else 0.0
                cluster_data.append([n, m, 'Test', 1, cluster_acc])
                if verbose:
                    print('%d: Cluster %d represents 1 with test accuracy of %.3f' % (n, m, cluster_acc))
            else:
                cluster_0_matches = np.count_nonzero(np.where(test_label==m,0,-1) == y_test_flat)
                test_correct_matches += cluster_0_matches
                cluster_acc = cluster_0_matches/cluster_sum if cluster_sum != 0 else 0.0
                cluster_data.append([n, m, 'Test', 0, cluster_acc])
                if verbose:
                    print('%d: Cluster %d represents 0 with test accuracy of %.3f' % (n, m, cluster_acc))
        test_acc = test_correct_matches/len(y_test_flat)
        test_accs.append(test_acc)
        if verbose:
            print('%d: Test Accuracy: %.3f' % (n, test_acc))
    
    plt.figure()
    plt.title("Silhouette Scores for %s using %s" % (dLabel, alg))
    plt.plot(list_clusters, sil_scores, 'o-', color="b")
    plt.xticks(list_clusters)
    plt.xlabel("No. of clusters")
    plt.ylabel("Silhouette Score")
    plt.savefig("charts/%s_%s_silScores.png" % (dLabel, alg))
    
    cluster_df = pd.DataFrame(cluster_data, columns=['N', 'Cluster', 'TrainTest', 'Rep', 'Accuracy'] )
    plt.figure()
    plt.title("%s Clustering Experiment for %s" % (alg, dLabel))
    mask_train_1 = (cluster_df['TrainTest']=='Train') & (cluster_df['Rep']==1)
    mask_train_0 = (cluster_df['TrainTest']=='Train') & (cluster_df['Rep']==0)
    mask_test_1 = (cluster_df['TrainTest']=='Test') & (cluster_df['Rep']==1)
    mask_test_0 = (cluster_df['TrainTest']=='Test') & (cluster_df['Rep']==0)
    plt.scatter(cluster_df[mask_train_1]['N'], cluster_df[mask_train_1]['Accuracy'], marker='+', color="b")
    plt.scatter(cluster_df[mask_train_0]['N'], cluster_df[mask_train_0]['Accuracy'], marker='_', color="b")
    plt.scatter(cluster_df[mask_test_1]['N'], cluster_df[mask_test_1]['Accuracy'], marker='+', color="r")
    plt.scatter(cluster_df[mask_test_0]['N'], cluster_df[mask_test_0]['Accuracy'], marker='_', color="r")
    plt.xticks(list_clusters)
    plt.legend(['Train Cluster representing 1','Train Cluster representing 0',
                'Test Cluster representing 1','Test Cluster representing 0'])
    plt.xlabel("No. of clusters")
    plt.ylabel("Cluster Accuracy")
    plt.savefig("charts/%s_%s_clusterExp.png" % (dLabel, alg))
    
    end_time = time.time()
    print('Run %s %s completed in %.3f' % (dLabel, alg, end_time-start_time))
    

def generatePCAResults(df1_split, n, dLabel='Banana', alg='PCA'):
    list_n = list(np.arange(1,n+1))

    pca = PCA(n_components=n, random_state=1337)
    x = pca.fit_transform(df1_split[0])
    
    plt.figure()
    plt.title("%s: Variance Explained for %s" % (alg, dLabel))
    plt.bar(list_n, pca.explained_variance_)
    plt.xticks(list_n)
    plt.xlabel("Component")
    plt.ylabel("Variance Explained")
    plt.savefig("charts/%s_%s_exp_var.png" % (dLabel, alg))
    generateReconstructionMSE(df1_split, n=n, dLabel=dLabel, alg=alg)

def generateICAResults(df1_split, n=11, dLabel='Banana', alg='ICA'):
    list_n = list(np.arange(1,n))
    list_mean_kurts = []
    list_errors = []
    
    plt.figure()
    plt.title("%s: Kurtosis for %s" % (alg, dLabel))
    plt.xticks(list_n)
    plt.xlabel("No. of components")
    plt.ylabel("Kurtosis")

    for n in list_n:
        ica = FastICA(n_components=n, max_iter=500, random_state=1337)
        x = ica.fit_transform(df1_split[0])
        list_kurts = []
        for m in range(x.shape[1]):
            kurt = abs(kurtosis(x[:,m]))
            list_kurts.append(kurt)
        mean_kurt = np.mean(list_kurts)
        error = np.std(list_kurts)
        list_mean_kurts.append(mean_kurt)
        list_errors.append(error)
        
    plt.xticks(list_n)
    plt.errorbar(list_n,list_mean_kurts, yerr=list_errors, marker = 'o')
    plt.savefig("charts/%s_%s_kurtosis.png" % (dLabel, alg))
    generateReconstructionMSE(df1_split, n=n, dLabel=dLabel, alg=alg)
    
def generateRPResults(df_split, **kwargs):
    generateReconstructionMSE(df_split, alg='RP', **kwargs)
    
def generateUMAPResults(df_split, list_nn, dLabel='Banana', alg='UMAP'):
    x_train, x_test, y_train, y_test = df_split
    
    fig, axs = plt.subplots(2, 2)
    mapper = umap.UMAP(n_neighbors = list_nn[0]).fit(x_train)
    umap.plot.points(mapper, labels=y_train.flatten(), ax=axs[0,0])
    
    mapper = umap.UMAP(n_neighbors = list_nn[1]).fit(x_train)
    umap.plot.points(mapper, labels=y_train.flatten(), ax=axs[0,1])
    
    mapper = umap.UMAP(n_neighbors = list_nn[2]).fit(x_train)
    umap.plot.points(mapper, labels=y_train.flatten(), ax=axs[1,0])
    
    mapper = umap.UMAP(n_neighbors = list_nn[3]).fit(x_train)
    umap.plot.points(mapper, labels=y_train.flatten(), ax=axs[1,1])
    fig.suptitle("%s: Projection for %s" % (alg, dLabel))
    fig.savefig("charts/%s_%s_umap_projection.png" % (dLabel, alg))
    
    # inverse_transform is expensive on UMAP
    #generateReconstructionMSE(df_split, alg='UMAP', **kwargs)
    
def generateReconstructionMSE(df_split, alg='', n=10, dLabel='Banana'):
    x_train = df_split[0]
    x_test = df_split[1]
    
    list_n = list(np.arange(1,n+1)) if alg not in ['UMAP'] else list(np.arange(2,n+2))
    list_train_mse_mean = []
    list_train_mse_std = []
    list_test_mse_mean = []
    list_test_mse_std = []
    
    for n in list_n:
        if alg == 'RP':
            dr = random_projection.SparseRandomProjection(n_components=n, random_state=1337)
        elif alg == 'PCA':
            dr = PCA(n_components=n, random_state=1337)
        elif alg =='ICA':
            dr = FastICA(n_components=n, max_iter=500, random_state=1337)
        elif alg == 'UMAP':
            dr = UMAP(n_components=n, random_state=1337)
        else:
            raise Exception('Model %s not supported' % alg)
        
        x_train_trf = dr.fit_transform(x_train)
        x_test_trf = dr.transform(x_test)
        x_train_inv_trf = dr.inverse_transform(x_train_trf)
        x_test_inv_trf = dr.inverse_transform(x_test_trf)
        
        list_train_mse = []
        list_test_mse = []
        for m in range(x_train.shape[1]):
            train_mse = mean_squared_error(x_train[:,m], x_train_inv_trf[:,m])
            test_mse = mean_squared_error(x_test[:,m], x_test_inv_trf[:,m])
            list_train_mse.append(train_mse)
            list_test_mse.append(test_mse)
        list_train_mse_mean.append(np.mean(list_train_mse))
        list_train_mse_std.append(np.std(list_train_mse))
        list_test_mse_mean.append(np.mean(list_test_mse))
        list_test_mse_std.append(np.std(list_test_mse))
    
    plt.figure()
    plt.title("%s: Reconstruction MSE for %s" % (alg, dLabel))
    plt.xticks(list_n)
    plt.xlabel("No. of components")
    plt.ylabel("MSE")
    plt.errorbar(list_n,list_train_mse_mean, yerr=list_train_mse_std, marker = 'o', color="b")
    plt.errorbar(list_n,list_test_mse_mean, yerr=list_test_mse_std, marker = 'o', color="r")
    plt.legend(['Train MSE','Test MSE'])
    plt.savefig("charts/%s_%s_reconstruction_mse.png" % (dLabel, alg))
    
    
def plotProjection( X, alg='EM', dLabel='Banana' ):
    # # https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/
    # X is a 2d numpy array
    plt.figure()
    b = X.shape[1]
    for i in range(0, 4):
        for j in range(0, 4):
            if i == j:
                continue
            x = X[:,i]
            y = X[:,j]
            a, b = np.polyfit(x, y, 1)
            plt.plot(x, a*x + b)
            plt.scatter(x, y, marker='.')
    plt.title("%s: Projection for %s" % (alg, dLabel))
    plt.xlabel("x")
    plt.ylabel("y")   
    plt.savefig("charts/%s_%s_projection.png" % (dLabel, alg))