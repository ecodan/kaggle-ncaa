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


def lookup_result(season, team1, team2, preds):
    # print('DIAG s=' + season + '|t1=' + str(team1) + '|t2=' + str(team2))
    key = ''
    if team1 < team2:
        key = season + '_' + str(team1) + '_' + str(team2)
        p = preds[preds['key'] == key]['prediction(label)'].values[0]
        if p == True:
            return team1
        else:
            return team2
    else:
        key = season + '_' + str(team2) + '_' + str(team1)
        p = preds[preds['key'] == key]['prediction(label)'].values[0]
        if p == True:
            return team2
        else:
            return team1

def lookup_proba(season, team1, team2, preds):
    # print('DIAG s=' + season + '|t1=' + str(team1) + '|t2=' + str(team2))
    key = ''
    if team1 < team2:
        key = season + '_' + str(team1) + '_' + str(team2)
    else:
        key = season + '_' + str(team2) + '_' + str(team1)
    # print('DIAG proba=' + str(preds[preds['key'] == key]['confidence(true)'].values[0]))
    return np.abs((preds[preds['key'] == key]['confidence(true)'].values[0] - 0.5) * 2)

def team_seed(seed,seeds):
    print('DIAG seed=' + str(seed))
    t = seeds[seeds['seed'] == seed]['team']
    print('DIAG t=' + str(t))
    return t.values[0]


def main(in_dir, out_dir='.'):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # preds = pd.read_table(in_dir + '/out/submission.csv', sep=',')
    preds = pd.read_table(in_dir + '/out/predictions02.csv', sep=',')

    slots = pd.read_table(in_dir + '/tourney_slots.csv', sep=',')
    slots = slots[slots['season'] == 'S']

    seeds = pd.read_table(in_dir + '/tourney_seeds.csv', sep=',')
    seeds = seeds[seeds['season'] == 'S']

    teams = pd.read_table(in_dir + '/teams.csv', sep=',')

    slots['team1'] = 0
    slots['team2'] = 0
    slots['wteam'] = 0
    slots['conf'] = 0.0
    slots['team1_name'] = ''
    slots['team2_name'] = ''
    slots['wteam_name'] = ''

    dfb = pd.DataFrame(columns=slots.columns)

    # process each round
    # pre-tourn
    r0 = slots['slot'].str.startswith('R') == False
    rslots = slots[r0]
    rslots['team1'] = rslots['strongseed'].apply(lambda x: team_seed(x, seeds))
    rslots['team2'] = rslots['weakseed'].apply(lambda x: team_seed(x, seeds))
    rslots['team1_name'] = rslots['team1'].apply(lambda x: teams[teams['id'] == x]['name'].values[0])
    rslots['team2_name'] = rslots['team2'].apply(lambda x: teams[teams['id'] == x]['name'].values[0])
    for i in rslots.index:
        rslots['wteam'][i] = lookup_result('S', rslots.ix[i]['team1'], rslots.ix[i]['team2'], preds)
        rslots['conf'][i] = lookup_proba('S', rslots.ix[i]['team1'], rslots.ix[i]['team2'], preds)
        rslots['wteam_name'][i] = teams[teams['id'] == rslots.ix[i]['wteam']]['name'].values[0]
    dfb = dfb.append(rslots)

    print ('DIAG dfb ' + str(dfb.head(50)))

    # round1
    r1 = slots['slot'].str.startswith('R1') == True
    rslots = slots[r1]
    rslots['team1'] = rslots['strongseed'].apply(lambda x: dfb[dfb['slot'] == x]['wteam'].values[0] if len(dfb[dfb['slot'] == x]) > 0 else seeds[seeds['seed'] == x]['team'].values[0])
    rslots['team2'] = rslots['weakseed'].apply(lambda x: dfb[dfb['slot'] == x]['wteam'].values[0] if len(dfb[dfb['slot'] == x]) > 0 else seeds[seeds['seed'] == x]['team'].values[0])
    rslots['team1_name'] = rslots['team1'].apply(lambda x: teams[teams['id'] == x]['name'].values[0])
    # print ('DIAG rslots ' + str(rslots.head(50)))
    rslots['team2_name'] = rslots['team2'].apply(lambda x: teams[teams['id'] == x]['name'].values[0])
    for i in rslots.index:
        rslots['wteam'][i] = lookup_result('S', rslots.ix[i]['team1'], rslots.ix[i]['team2'], preds)
        rslots['conf'][i] = lookup_proba('S', rslots.ix[i]['team1'], rslots.ix[i]['team2'], preds)
        rslots['wteam_name'][i] = teams[teams['id'] == rslots.ix[i]['wteam']]['name'].values[0]
    dfb = dfb.append(rslots)

    print ('DIAG dfb ' + str(dfb.head(50)))

    for r in range(2,7):
        rn = slots['slot'].str.startswith('R' + str(r)) == True
        rslots = slots[rn]
        rslots['team1'] = rslots['strongseed'].apply(lambda x: dfb[dfb['slot'] == x]['wteam'].values[0] if len(dfb[dfb['slot'] == x]) > 0 else -1)
        rslots['team2'] = rslots['weakseed'].apply(lambda x: dfb[dfb['slot'] == x]['wteam'].values[0] if len(dfb[dfb['slot'] == x]) > 0 else -1)
        rslots['team1_name'] = rslots['team1'].apply(lambda x: teams[teams['id'] == x]['name'].values[0])
        # print ('DIAG rslots ' + str(rslots.head(50)))
        rslots['team2_name'] = rslots['team2'].apply(lambda x: teams[teams['id'] == x]['name'].values[0])
        for i in rslots.index:
            rslots['wteam'][i] = lookup_result('S', rslots.ix[i]['team1'], rslots.ix[i]['team2'], preds)
            rslots['conf'][i] = lookup_proba('S', rslots.ix[i]['team1'], rslots.ix[i]['team2'], preds)
            rslots['wteam_name'][i] = teams[teams['id'] == rslots.ix[i]['wteam']]['name'].values[0]
        dfb = dfb.append(rslots)

    print ('DIAG dfb ' + str(dfb.head(50)))

    dfb.to_csv(out_dir + "/brackets02.csv")


if __name__=='__main__':

    args = { 'in_dir':  '/Users/dan/dev/datasci/kaggle/ncaa/',
             'out_dir': '/Users/dan/dev/datasci/kaggle/ncaa/out/'}
    model = main(**args)

#    print "args=" + str(sys.argv[0:])
#    model = main(sys.argv[1], sys.argv[2])
