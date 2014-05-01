__author__ = 'dan'

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

def mktrain(dft, dfd, dfs):

    merged = []

    for s in np.unique(dft['season']):
        print("season: " + s)
        dfts = dft[dft['season'] == s]
        dfss = dfs[dfs['season'] == s]
        dfds = dfd[dfd['season'] == s]
        for row in dfts.as_matrix():
            wsos = dfss[dfss['team'] == row[2]]
            lsos = dfss[dfss['team'] == row[4]]
            wseed = dfds[dfds['team'] == row[2]]
            lseed = dfds[dfds['team'] == row[4]]
            wseednum = wseed.iloc[0]['seed'][1:3]
            lseednum = lseed.iloc[0]['seed'][1:3]

            if (len(merged) == 0):
                merged = np.array([row[0],row[1],row[2],row[3],row[4],row[5],row[6],wseednum,wsos.iloc[0]['won'],wsos.iloc[0]['lost'],wsos.iloc[0]['wp'],wsos.iloc[0]['owp'],wsos.iloc[0]['oowp'],wsos.iloc[0]['sos'],wsos.iloc[0]['rpi'],lseednum,lsos.iloc[0]['won'],lsos.iloc[0]['lost'],lsos.iloc[0]['wp'],lsos.iloc[0]['owp'],lsos.iloc[0]['oowp'],lsos.iloc[0]['sos'],lsos.iloc[0]['rpi']])
            else:
                merged = np.vstack(( merged, np.array([row[0],row[1],row[2],row[3],row[4],row[5],row[6],wseednum,wsos.iloc[0]['won'],wsos.iloc[0]['lost'],wsos.iloc[0]['wp'],wsos.iloc[0]['owp'],wsos.iloc[0]['oowp'],wsos.iloc[0]['sos'],wsos.iloc[0]['rpi'],lseednum,lsos.iloc[0]['won'],lsos.iloc[0]['lost'],lsos.iloc[0]['wp'],lsos.iloc[0]['owp'],lsos.iloc[0]['oowp'],lsos.iloc[0]['sos'],lsos.iloc[0]['rpi']]) ))


    dfm = pd.DataFrame(data=merged, columns=['season','daynum','wteam','wscore','lteam','lscore','numot','w_seed','w_won','w_lost','w_wp','w_owp','w_oowp','w_sos','w_rpi','l_seed','l_won','l_lost','l_wp','l_owp','l_oowp','l_sos','l_rpi'])
    return dfm

def mktest(dft, dfd, dfs):

    merged = []

    for s in ['N','O','P','Q','R']:
        print("season: " + s)
        dfts = dft[dft['season'] == s]
        dfss = dfs[dfs['season'] == s]
        dfds = dfd[dfd['season'] == s]
        for row in dfts.as_matrix():
            wsos = dfss[dfss['team'] == row[2]]
            lsos = dfss[dfss['team'] == row[4]]
            wseed = dfds[dfds['team'] == row[2]]
            lseed = dfds[dfds['team'] == row[4]]
            wseednum = wseed.iloc[0]['seed'][1:3]
            lseednum = lseed.iloc[0]['seed'][1:3]
            if (len(merged) == 0):
                merged = np.array([row[0],row[1],wseednum,wsos.iloc[0]['won'],wsos.iloc[0]['lost'],wsos.iloc[0]['wp'],wsos.iloc[0]['owp'],wsos.iloc[0]['oowp'],wsos.iloc[0]['sos'],wsos.iloc[0]['rpi'],lseednum,lsos.iloc[0]['won'],lsos.iloc[0]['lost'],lsos.iloc[0]['wp'],lsos.iloc[0]['owp'],lsos.iloc[0]['oowp'],lsos.iloc[0]['sos'],lsos.iloc[0]['rpi']])
            else:
                merged = np.vstack(( merged, np.array([row[0],row[1],wseednum,wsos.iloc[0]['won'],wsos.iloc[0]['lost'],wsos.iloc[0]['wp'],wsos.iloc[0]['owp'],wsos.iloc[0]['oowp'],wsos.iloc[0]['sos'],wsos.iloc[0]['rpi'],lseednum,lsos.iloc[0]['won'],lsos.iloc[0]['lost'],lsos.iloc[0]['wp'],lsos.iloc[0]['owp'],lsos.iloc[0]['oowp'],lsos.iloc[0]['sos'],lsos.iloc[0]['rpi']]) ))


    dfm = pd.DataFrame(data=merged, columns=['season','daynum','w_seed','w_won','w_lost','w_wp','w_owp','w_oowp','w_sos','w_rpi','l_seed','l_won','l_lost','l_wp','l_owp','l_oowp','l_sos','l_rpi'])
    return dfm


def mksubmission(dfd, dfs):

    merged = []

    for s in ['S']:
        print("season: " + s)
        dfss = dfs[dfs['season'] == s]
        dfds = dfd[dfd['season'] == s]
        teams = np.sort(np.unique(dfds['team']))
        for t in teams:
            for u in teams:
                if (u <= t):
                    continue

                wsos = dfss[dfss['team'] == t]
                lsos = dfss[dfss['team'] == u]
                wseed = dfds[dfds['team'] == t]
                lseed = dfds[dfds['team'] == u]
                wseednum = wseed.iloc[0]['seed'][1:3]
                lseednum = lseed.iloc[0]['seed'][1:3]
                key = s + '_' + str(t) + '_' + str(u)
                if (len(merged) == 0):
                    merged = np.array([key,s,wseednum,wsos.iloc[0]['won'],wsos.iloc[0]['lost'],wsos.iloc[0]['wp'],wsos.iloc[0]['owp'],wsos.iloc[0]['oowp'],wsos.iloc[0]['sos'],wsos.iloc[0]['rpi'],lseednum,lsos.iloc[0]['won'],lsos.iloc[0]['lost'],lsos.iloc[0]['wp'],lsos.iloc[0]['owp'],lsos.iloc[0]['oowp'],lsos.iloc[0]['sos'],lsos.iloc[0]['rpi']])
                else:
                    merged = np.vstack(( merged, np.array([key,s,wseednum,wsos.iloc[0]['won'],wsos.iloc[0]['lost'],wsos.iloc[0]['wp'],wsos.iloc[0]['owp'],wsos.iloc[0]['oowp'],wsos.iloc[0]['sos'],wsos.iloc[0]['rpi'],lseednum,lsos.iloc[0]['won'],lsos.iloc[0]['lost'],lsos.iloc[0]['wp'],lsos.iloc[0]['owp'],lsos.iloc[0]['oowp'],lsos.iloc[0]['sos'],lsos.iloc[0]['rpi']]) ))


    dfm = pd.DataFrame(data=merged, columns=['key','season','w_seed','w_won','w_lost','w_wp','w_owp','w_oowp','w_sos','w_rpi','l_seed','l_won','l_lost','l_wp','l_owp','l_oowp','l_sos','l_rpi'])
    return dfm


def mktrain2(dft, dfd, dfs):
    # split into 2
    dft2 = dft.sort(['wscore'])
    dft2a = dft2.head(len(dft2)/2)
    dft2b = dft2.tail(len(dft2)/2)

    # merge team stats on winning team
    dfma = dft2a.merge(dfs, left_on=['wteam','season'], right_on=['team','season'])
    dfmb = dft2b.merge(dfs, left_on=['lteam','season'], right_on=['team','season'])
    print ('dfma shape 1 ' + str(dfma.shape))
    print ('dfmb shape 1 ' + str(dfmb.shape))

    # merge team stats on losing team
    dfma = dfma.merge(dfs, left_on=['lteam','season'], right_on=['team','season'], suffixes=['_t1','_t2'])
    dfmb = dfmb.merge(dfs, left_on=['wteam','season'], right_on=['team','season'], suffixes=['_t1','_t2'])
    print ('dfma shape 2 ' + str(dfma.shape))
    print ('dfmb shape 2 ' + str(dfmb.shape))

    # merge seed on winning team
    dfma = dfma.merge(dfd, left_on=['wteam', 'season'], right_on=['team','season'])
    dfmb = dfmb.merge(dfd, left_on=['lteam', 'season'], right_on=['team','season'])
    print ('dfma shape 3 ' + str(dfma.shape))
    print ('dfmb shape 3 ' + str(dfmb.shape))

    # merge seed on losing team
    dfma = dfma.merge(dfd, left_on=['lteam', 'season'], right_on=['team','season'], suffixes=['x_t1','x_t2'])
    dfmb = dfmb.merge(dfd, left_on=['wteam', 'season'], right_on=['team','season'], suffixes=['x_t1','x_t2'])
    print ('dfma shape 4 ' + str(dfma.shape))
    print ('dfmb shape 4 ' + str(dfmb.shape))

    dfma['match'] = dfma[['apf_t1','apf_t2']].apply(lambda x: x['apf_t1']-x['apf_t2'], axis=1)
    dfmb['match'] = dfma[['apf_t1','apf_t2']].apply(lambda x: x['apf_t1']-x['apf_t2'], axis=1)

    dfma['label'] = 1
    dfmb['label'] = 0
    return dfma, dfmb


def mksubmission2(dfd, dfs):

    print('mksub 2')
    print ("dfs " + str(dfs.shape) + "|\n" + str(dfs.head(10)))
    print ("dfd " + str(dfd.shape) + "|\n" + str(dfd.head(10)))

    merged = []
    # for s in ['N','O','P','Q','R']:
    for s in ['S']:
        print("season: " + s)
        dfss = dfs[dfs['season'] == s]
        dfds = dfd[dfd['season'] == s]

        # get all teams in the tournament
        teams = np.sort(np.unique(dfds['team']))
        for t in teams:
            for u in teams:
                if (u <= t):
                    continue

                key = s + '_' + str(t) + '_' + str(u)
                if (len(merged) == 0):
                    merged = np.array([key,s,t,u])
                else:
                    merged = np.vstack(( merged, np.array([key,s,t,u]) ))


    dfm = pd.DataFrame(data=merged, columns=['key','season','team1','team2'])
    print ("dfm " + str(dfm.shape) + "|\n" + str(dfm.head(10)))
    dfm['team1'] = dfm['team1'].astype('int')
    dfm['team2'] = dfm['team2'].astype('int')

    # append SOS to each team for each pairing
    dfma = dfm.merge(dfs, left_on=['team1','season'], right_on=['team','season'])
    dfma = dfma.merge(dfs, left_on=['team2','season'], right_on=['team','season'], suffixes=['_t1','_t2'])
    print ("dfma 1 " + str(dfma.shape) + "|\n" + str(dfma.head(10)))

    # append seeds to each team for each pairing
    # dfd1 = dfd[dfd.columns.values]
    # dfd1.columns = ['season', 'seed_t1',  'team_t1', 'region_t1', 'seednum_t1']
    # dfd2 = dfd[dfd.columns.values]
    # dfd2.columns = ['season', 'seed_t2',  'team_t2', 'region_t2', 'seednum_t2']
    dfma = dfma.merge(dfd, left_on=['team1', 'season'], right_on=['team','season'])
    # print ("dfma 2a " + str(dfma.shape) + "|\n" + str(dfma.head(10)))
    # dfma = dfma.merge(dfd2, left_on=['team2', 'season'], right_on=['team_t2','season'])
    dfma = dfma.merge(dfd, left_on=['team2', 'season'], right_on=['team','season'], suffixes=['x_t1','x_t2'])
    dfma['match'] = dfma[['apf_t1','apf_t2']].apply(lambda x: x['apf_t1']-x['apf_t2'], axis=1)

    dfma['act'] = -1
    print ("dfma 2 " + str(dfma.shape) + "|\n" + str(dfma.head(10)))

    # for i in range(0,len(dft)):
    #     print('compare ' + str(dft.iloc[i]['wteam']) + ' & ' + str(dft.iloc[i]['lteam']) + ' s=' + str(dft.iloc[i]['season']) )
    #     dfgw = dfma['act'][ (dft.iloc[i]['wteam'] == dfma['team1']) & (dft.iloc[i]['lteam'] == dfma['team2']) & (dft.iloc[i]['season'] == dfma['season']) ] = 1
    #     # if len(dfgw):
    #     #     dfma.ix[dfgw.ix] = 1
    #     dfgl = dfma['act'][(dft.iloc[i]['lteam'] == dfma['team1']) & (dft.iloc[i]['wteam'] == dfma['team2']) & (dft.iloc[i]['season'] == dfma['season'])] = 0
    #     # if len(dfgl):
    #     #     dfma.ix[dfgw.ix] = -1

    return dfma


def loaddata(in_file):
    print "reading datafile: " + str(in_file)
    df = pd.read_table(in_file, sep=',')
    return df


def main(tourn_file, seed_file, sos_file, out_dir='.'):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dfs = loaddata(sos_file)
    dft = loaddata(tourn_file)
    dfd = loaddata(seed_file)
    dfd['region'] = dfd['seed'].apply(lambda x: x[0:1])
    dfd['seednum'] = dfd['seed'].apply(lambda x: x[1:3])

    dfa, dfb = mktrain2(dft, dfd, dfs)
    dfa.to_csv(out_dir + '/train02_1.csv',index=False)
    dfb.to_csv(out_dir + '/train02_2.csv',index=False)
    dfboth = dfa.append(dfb)
    dfboth.to_csv(out_dir + '/train02.csv',index=False)

    #
    # df = mktest(dft, dfd, dfs)
    # df.to_csv(out_dir + '/test.csv',index=False)

    df = mksubmission2(dfd, dfs)
    df.to_csv(out_dir + '/sub02.csv',index=False)
    # df = mksubmission(dfd, dfs)
    # df.to_csv(out_dir + '/sub.csv',index=False)


if __name__=='__main__':

    args = { 'tourn_file':  '/Users/dan/dev/datasci/kaggle/ncaa/tourney_results.csv',
             'seed_file':  '/Users/dan/dev/datasci/kaggle/ncaa/tourney_seeds.csv',
             'sos_file':  '/Users/dan/dev/datasci/kaggle/ncaa/out/sos_out.csv',
             'out_dir': '/Users/dan/dev/datasci/kaggle/ncaa/out/'}
    model = main(**args)

#    print "args=" + str(sys.argv[0:])
#    model = main(sys.argv[1], sys.argv[2])
