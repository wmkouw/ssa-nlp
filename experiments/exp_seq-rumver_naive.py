#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment on classifying rumour veracity sequentially.

In this script, I use a naive classifier that receives rumour veracity data
from previous days and predicts veracity on the next day.

Author: W.M. Kouw
Date: 15-10-2018
"""
import numpy as np
import pandas as pd
import pickle as pc

from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer


# Preallocate performance array
perf_array = []
days_array = []
rums_array = []

# Load data
tweets = pd.read_csv('../data/PHEME/PHEME.csv', sep='\t')

# Select rumours
rumours = np.unique(tweets['which_rumour'].astype('str'))

for r, rumour in enumerate(rumours):

    # Select tweets
    tweets_r = tweets[tweets['which_rumour'] == rumours[0]]

    # Map data to bag-of-words format
    X = CountVectorizer().fit_transform(tweets_r['content']).toarray()
    Y = tweets_r['misinformation'].as_matrix()

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

        # Define classifier
        clf = linear_model.LogisticRegressionCV()

        # Train classifier on data from current and previous days
        clf.fit(trn_X, trn_Y)

        # Test on data from current day and store
        perf_array.append(clf.score(tst_X, tst_Y))

        # Store day and rumour
        days_array.append(days[d])
        rums_array.append(rumour)

# Compact to DataFrame
performance = pd.DataFrame({'acc': perf_array,
                            'days': days_array,
                            'rumours': rums_array})

# Write to file
performance.to_csv('./results/seq-rumver_naive.csv')
