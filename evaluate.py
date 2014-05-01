__author__ = 'dan'

__author__ = 'dan'

import math
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
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model

def cat_and_clean(root, dirs, out_dir):
    print ("done with cat and clean")

def strip_junk(df):
    #print y
    return df

def timedelta(x):
    i = None
    if (x[0] > x[1]):
        i = x[0] - x[1]
    else:
        i = x[1] - x[0]
    secs = i.total_seconds()
    return secs*1000

def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]


def clean(df):

    return df


def evaluate(df):
    dfs = df[df['act'] >= 0]
    dfs['confidence(true)'][dfs['confidence(true)'] == 0] = 0.000001
    dfs['confidence(true)'][dfs['confidence(true)'] == 1] = 0.999999
    dfs['score'] = dfs.apply(lambda x: -((x['act']*math.log(x['confidence(true)']))+(1-x['act'])*(math.log(1-x['confidence(true)']))), axis=1)

    return np.sum(dfs['score'])/len(dfs)

def loaddata(in_file):
    print "reading datafile: " + str(in_file)
    df = pd.read_table(in_file, sep=',')
    return df

def main(in_file, out_dir='.'):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = loaddata(in_file)
    df = clean(df)
    score = evaluate(df)
    print(str(score))

if __name__=='__main__':

    args = { 'in_file':  '/Users/dan/dev/datasci/kaggle/ncaa/out/submission02.csv',
             'out_dir': '/Users/dan/dev/datasci/kaggle/ncaa/out/'}
    model = main(**args)

#    print "args=" + str(sys.argv[0:])
#    model = main(sys.argv[1], sys.argv[2])
