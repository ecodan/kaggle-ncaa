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


def calculate_for_period(df, s, daystart, dayend):

    print ('calculate for period ' )
    sos = []
    wl = []

    dfs = df[df['daynum'] > daystart]
    dfs = dfs[dfs['daynum'] <= dayend]

    for t in np.unique(np.concatenate((np.unique(dfs['wteam']), np.unique(dfs['lteam'])))):
        won = len(np.where((dfs['wteam'] == t) & (dfs['wloc'] == 'A'))[0])*1.4 + len(np.where((dfs['wteam'] == t) & (dfs['wloc'] == 'N'))[0]) + len(np.where((dfs['wteam'] == t) & (dfs['wloc'] == 'H'))[0])*0.6
        lost = len(np.where((dfs['lteam'] == t) & (dfs['wloc'] == 'H'))[0])*1.4 + len(np.where((dfs['lteam'] == t) & (dfs['wloc'] == 'N'))[0]) + len(np.where((dfs['lteam'] == t) & (dfs['wloc'] == 'A'))[0])*0.6
        # print(t,':',won,'-',lost)
        wins = dfs[dfs['wteam'] == t]
        losses = dfs[dfs['lteam'] == t]
        pf = wins['wscore'].sum() + losses['lscore'].sum()
        apf = (pf)*1.0/(len(wins) + len(losses))
        apa = (losses['wscore'].sum() + wins['lscore'].sum())*1.0/(len(wins) + len(losses))
        wp = 0.0
        if (won+lost) > 0:
            wp = float(won)/(won+lost)
        if (len(wl) == 0):
            wl = np.array([t,won,lost,wp,0.0,0.0,apf,apa])
        else:
            wl = np.vstack((wl,[t,won,lost,wp,0.0,0.0,apf,apa]))

    # calculate opp
    dfwl = pd.DataFrame(data=wl,columns=['team','won','lost','wp','owp','oowp','apf','apa'])
    # print(dfwl.head(10))
    wl2 = []
    for row in wl:
        opp = np.concatenate((np.unique(dfs[dfs['wteam'] == row[0]]['lteam']).values,np.unique(dfs[dfs['lteam'] == row[0]]['wteam']).values))
        dfopp = dfwl[dfwl['team'].isin(opp)]
        won = dfopp['won'].sum()
        lost = dfopp['lost'].sum()
        wp = 0.0
        if (won + lost) > 0:
            wp = float(won)/(won+lost)
        #print(wp)
        if (len(wl2) == 0):
            wl2 = np.array([row[0],row[1],row[2],row[3],wp,0.0,row[6],row[7]])
        else:
            wl2 = np.vstack((wl2,np.array([row[0],row[1],row[2],row[3],wp,0.0,row[6],row[7]])))

    # calculate opp-opp
    dfwl = pd.DataFrame(wl2,columns=['team','won','lost','wp','owp','oowp','apf','apa'])
    # print(dfwl.head(10))
    wl3 = []
    for row in wl2:
        opp = np.concatenate((np.unique(dfs[dfs['wteam'] == row[0]]['lteam']).values,np.unique(dfs[dfs['lteam'] == row[0]]['wteam']).values))
        dfopp = dfwl[dfwl['team'].isin(opp)]
        if (len(wl3) == 0):
            wl3 = np.array([row[0],row[1],row[2],row[3],row[4],dfopp['owp'].mean(),row[6],row[7]])
        else:
            wl3 = np.vstack((wl3, np.array([row[0],row[1],row[2],row[3],row[4],dfopp['owp'].mean(),row[6],row[7]])))

    for row in wl3:
        str = ((2*row[4]) + row[5])/3.0
        rpi = (row[3] * 0.25) + str
        if (len(sos) == 0):
            sos = np.array([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],str,rpi])
        else:
            sos = np.vstack((sos,np.array([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],str,rpi])))

    dfsos = pd.DataFrame(sos, columns=['team','won','lost','wp','owp','oowp','apf','apa','sos','rpi'])
    dfsos['season'] = s
    dfsos = dfsos[['team','season','won','lost','wp','owp','oowp','apf','apa','sos','rpi']]
    # need to convert to int but first to float for some reason
    dfsos['team'] = dfsos['team'].astype(np.dtype('f8'))
    dfsos['team'] = dfsos['team'].astype(np.dtype('i4'))
    # print(dfsos.head(10))
    return dfsos


def calculate(df):

    print ('starting calculate...')
    dfsos = pd.DataFrame(columns=['team','season','won','lost','wp','owp','oowp','apf','apa','sos','rpi', 'won1','lost1','wp1','owp1','oowp1','apf1','apa1','sos1','rpi1','won2','lost2','wp2','owp2','oowp2','apf2','apa2','sos2','rpi2','won3','lost3','wp3','owp3','oowp3','apf3','apa3','sos3','rpi3'])
    for s in np.unique(df['season']):
        # wl = np.array(['team','w','l','wp','owp','oowp'])
        wl = []
        print ('starting season ' + s)
        dfs = df[df['season'] == s]
        print('data len = ' + str(len(dfs)))

        # whole season
        dfw = calculate_for_period(dfs, s, 0, 136)
        print ('DIAG dfw ' + str(dfw.shape))
        df1 = calculate_for_period(dfs, s, 0, 50)
        print ('DIAG df1 ' + str(df1.shape) + '|\n' + str(df1.head(10)))
        df2 = calculate_for_period(dfs, s, 50, 100)
        print ('DIAG df2 ' + str(df2.shape) + '|\n' + str(df2.head(10)))
        df3 = calculate_for_period(dfs, s, 100, 136)
        print ('DIAG df3 ' + str(df3.shape) + '|\n' + str(df3.head(10)))

        # calculate top 25 wins
        print ('calc top 25')
        t25w = []
        dftop = dfw.sort(['rpi']).tail(25)
        print(dftop.describe())
        for t in np.unique(dfs['wteam']):
            dft = dfs[dfs['wteam'] == t]
            dftf = dft[dft['lteam'].isin(dftop['team'])]
            top25wins = len(dft[dft['lteam'].isin(dftop['team'])])
            if (len(t25w) == 0):
                t25w = np.array([t, top25wins])
            else:
                t25w = np.vstack((t25w,np.array([t, top25wins])))
        dt25 = pd.DataFrame(t25w, columns=['team','top25wins'])

        # join on all attributes
        print ('join all stats')
        print(dfw.describe())
        print(dt25.describe())

        print ('DIAG merge 1')
        dfa = dfw.merge(dt25, on=['team'], how='left')
        print ('DIAG merge 2; dfa.shape=' + str(dfa.shape) + '|\n' + str(dfa.head(10)))
        dfa = dfa.merge(df1, on=['team', 'season'])
        dfa.columns=['team','season','won','lost','wp','owp','oowp','apf','apa','sos','rpi','top25wins', 'won1','lost1','wp1','owp1','oowp1','apf1','apa1','sos1','rpi1']
        print ('DIAG merge 3; dfa.shape=' + str(dfa.shape) + '|\n' + str(dfa.head(10)))
        dfa = dfa.merge(df2, on=['team', 'season'])
        dfa.columns=['team','season','won','lost','wp','owp','oowp','apf','apa','sos','rpi','top25wins', 'won1','lost1','wp1','owp1','oowp1','apf1','apa1','sos1','rpi1','won2','lost2','wp2','owp2','apf2','apa2','oowp2','sos2','rpi2']
        print ('DIAG merge 4; dfa.shape=' + str(dfa.shape) + '|\n' + str(dfa.head(10)))
        dfa = dfa.merge(df3, on=['team', 'season'])
        print ('DIAG append; dfa.shape=' + str(dfa.shape))
        dfa.columns=['team','season','won','lost','wp','owp','oowp','apf','apa','sos','rpi','top25wins', 'won1','lost1','wp1','owp1','oowp1','apf1','apa1','sos1','rpi1','won2','lost2','wp2','owp2','oowp2','apf2','apa2','sos2','rpi2','won3','lost3','wp3','owp3','oowp3','apf3','apa3','sos3','rpi3']
        print(str(len(dfa)))
        # append to sos
        dfsos = dfsos.append(dfa)
        print('sos len=' + str(len(dfsos)))

    return dfsos



def loaddata(in_file):
    print "reading datafile: " + str(in_file)
    df = pd.read_table(in_file, sep=',')
    return df


def main(in_file, out_dir='.'):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print ('reading data')
    df = loaddata(in_file)
    print ('data shape = ' + str(df.shape))
    df = clean(df)
    df = calculate(df)
    df.to_csv(out_dir + '/sos_out.csv',index=False)


if __name__=='__main__':

    args = {
        'in_file':  '/Users/dan/dev/datasci/kaggle/ncaa/regular_season_results.csv',
        # 'in_file':  '/Users/dan/dev/datasci/kaggle/ncaa/rd2/regular_season_results_thru_S_day_132.csv',
        'out_dir': '/Users/dan/dev/datasci/kaggle/ncaa/out/'}
    model = main(**args)

#    print "args=" + str(sys.argv[0:])
#    model = main(sys.argv[1], sys.argv[2])
