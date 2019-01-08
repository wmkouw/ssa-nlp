#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment on classifying rumour veracity sequentially.

Author: W.M. Kouw
Date: 15-10-2018
"""
import numpy as np
import pandas as pd
import pickle as pc

import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

from libtlda.iw import ImportanceWeightedClassifier

# Toggle visualizations
viz = False

# Preallocate performance array
perf_n = []
perf_a = []
days_array = []
rums_array = []

# Load data
tweets = pd.read_csv('../data/PHEMEj.csv', sep='\t')

# Select rumours
rumours = np.unique(tweets['which_rumour'].astype('str'))

# Select subset of rumours
subset = [0, 3, 4, 7]

for rumour in rumours[subset]:

    # Select tweets
    tweets_r = tweets[tweets['which_rumour'] == rumour]

    # Map data to bag-of-words format
    X = CountVectorizer().fit_transform(tweets_r['content']).toarray()
    Y = tweets_r['misinformation'].as_matrix()

    # PCA on data
    X = PCA(n_components=3).fit_transform(X)

    if viz:
        fig, ax = plt.subplots()
        ax.scatter(X[Y == 0, 0], X[Y == 0, 1], color='r', label='y=0')
        ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color='b', label='y=1')
        ax.set_title(rumour)
        ax.set_xlabel('p_1')
        ax.set_ylabel('p_2')
        ax.legend()
        plt.show()

    # List of days on which was tweeted
    days = np.unique(tweets_r['created_date'])

    # Loop over days
    for d in range(1, len(days)):

        # Set range up to yesterday
        past = range(d)

        # Create training data from all previous days
        trn_index = tweets_r['created_date'].isin(days[past]).values.tolist()

        # Find all tweets from today
        tst_index = (tweets_r['created_date'] == days[d]).values.tolist()

        # Split out training data
        trn_X = X[trn_index, :]
        trn_Y = Y[trn_index]

        # Split out test data
        tst_X = X[tst_index, :]
        tst_Y = Y[tst_index]

        # Define classifiers
        clf_n = linear_model.LogisticRegression(C=0.1)
        clf_a = ImportanceWeightedClassifier(loss='logistic', l2=0.1)

        # Train classifier on data from current and previous days
        clf_n.fit(trn_X, trn_Y)
        clf_a.fit(trn_X, trn_Y, tst_X)

        # Make predictions
        preds_n = clf_n.predict(tst_X)
        preds_a = clf_a.predict(tst_X)

        # Test on data from current day and store
        perf_n.append(np.mean(preds_n != tst_Y))
        perf_a.append(np.mean(preds_a != tst_Y))

        # Store day and rumour
        days_array.append(days[d])
        rums_array.append(rumour)

# Compact to DataFrame
performance = pd.DataFrame({'err_n': perf_n,
                            'err_a': perf_a,
                            'days': days_array,
                            'rumours': rums_array})

# Write to file
performance.to_csv('./results/results_rumEval17.csv')
