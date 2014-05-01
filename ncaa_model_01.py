__author__ = 'dan'


import os
import sys
import xml.dom.minidom as xml
import pandas as pd
import numpy as np
import random
from pandas.core.series import Series
from pandas.tseries.index import date_range
import matplotlib as mp
import matplotlib.pyplot as plt
import pylab
import datetime
import csv
import sklearn as sl
from sklearn import neighbors
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Imputer

from sklearn import preprocessing as pre
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn.linear_model as lm


def mae(y_pred, y_act):
    return (np.abs(y_act - y_pred).sum() * 1.0)/len(y_pred)



def main(in_dir, out_dir):

    # read in training file
    print('reading train file...')
    df = pd.read_table(in_dir + '/out/train02.csv', sep=',')
    print ('training set size: ' + str(df.shape))

    # drop columns
    df = df.drop(['regionx_t2','teamx_t2','seedx_t2','regionx_t1','teamx_t1','seedx_t1','team_t1', 'team_t2'], axis=1)
    # print (df[df.columns[7::]])

    # clean
    imputer = Imputer()
    imputer.fit(df[df.columns[7:-1]])
    clean = imputer.transform(df[df.columns[7:-1]])

    # scale
    scaler = pre.StandardScaler()
    scaled = scaler.fit_transform(clean)
    dfs = pd.DataFrame(scaled, columns=df.columns[7:-1])
    dfs['label'] = df['label'].values

    # reduce for loan-default classification
    # model_cols = ['wp3_t1','top25wins_t1','sos3_t1','wp3_t2','top25wins_t2','sos3_t2'] # .60-.65
    # model_cols = ['match','wp2_t1','sos2_t1','wp2_t2','sos2_t2','wp3_t1','sos3_t1','wp3_t2','sos3_t2','top25wins_t1','top25wins_t2'] #.65-.66
    # model_cols = ['wp1_t1','sos1_t1','wp1_t2','sos1_t2','wp2_t1','sos2_t1','wp2_t2','sos2_t2','wp3_t1','sos3_t1','wp3_t2','sos3_t2','top25wins_t1','top25wins_t2'] #.60-.71
    # model_cols = ['match','apf3_t1','apf3_t2','wp2_t1','sos2_t1','wp2_t2','sos2_t2','wp3_t1','sos3_t1','wp3_t2','sos3_t2','top25wins_t1','top25wins_t2'] #.69-.72
    model_cols = dfs.columns[0:-1]

    # X = dfs[dfs.columns[0:-1]]
    X = dfs[model_cols]
    y = dfs[dfs.columns[-1]]

    lsvc = LinearSVC(C=0.1, penalty="l1", dual=False, verbose = 1)
    lsvc.fit(X, y)
    X = lsvc.transform(X)
    print('reduced X = ' + str(np.shape(X)))

    # train and x-validate
    #clf = LogisticRegression(C=1e20,penalty='l2')
    # clf = RandomForestClassifier(n_estimators=10,max_features=None)
    # clf = sl.ensemble.ExtraTreesClassifier(n_estimators=10,max_features=None)
    clf = svm.SVC(kernel='rbf',probability=True) # getting into the .8s

    print('xval\n' + str(cross_val_score(clf, X, y, cv=10)))

    X_train,X_test,y_train,y_test = train_test_split(X, y)

    clf.fit(X_train,y_train)
    z = clf.predict(X_test)
    print ('roc auc: ' + str(roc_auc_score(y_test,z)))
    print ('conf matrix:\n' + str(sl.metrics.confusion_matrix(y_test,z)))
    print ('class report:\n' + str(sl.metrics.classification_report(y_test,z)))
    # imp = pd.DataFrame(clf.feature_importances_, X.columns)
    # imp.to_csv(out_dir + '/feature_importances.csv')

    # retrain with all values (is this wise???)
    clf.fit(X,y)
    z2 = clf.predict(X)
    print ('roc auc (full): ' + str(roc_auc_score(y,z2)))


    #######################
    # predict
    print('reading train file...')
    df = pd.read_table(in_dir + '/out/sub02.csv', sep=',')
    # print (df)

    # drop columns
    df = df.drop(['regionx_t2','teamx_t2','seedx_t2','regionx_t1','teamx_t1','seedx_t1','team_t1', 'team_t2'], axis=1)
    # print (df[df.columns[1::]])

    # clean
    imputer = Imputer()
    imputer.fit(df[df.columns[4:-1]])
    clean = imputer.transform(df[df.columns[4:-1]])

    # scale
    scaled = scaler.transform(clean)
    dfs = pd.DataFrame(scaled, columns=df.columns[4:-1])
    # print (dfs)

    # classification
    # X = dfs[dfs.columns]
    X = dfs[model_cols]
    X = lsvc.transform(X)
    y = clf.predict(X)
    dfs['prediction(label)'] = y
    # print(clf.predict_proba(X)[::,0])
    dfs['confidence(true)'] = clf.predict_proba(X)[::,1]
    dfs['confidence(false)'] = clf.predict_proba(X)[::,0]
    dfs['key'] = df['key']
    dfs.to_csv(out_dir + '/predictions02.csv')

    dfp = dfs[['key', 'confidence(true)']]
    dfp.columns = ['id','pred']
    dfp.to_csv(out_dir + '/submission02.csv', index=False)

if __name__=='__main__':

    args = { 'in_dir':  '/Users/dan/dev/datasci/kaggle/ncaa',
             'out_dir': '/Users/dan/dev/datasci/kaggle/ncaa/out'}
    model = main(**args)
