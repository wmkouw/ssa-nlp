"""
Read training data from RumourEval 2019.

RumourEval is a shared task in rumour stance classification. More info at:
https://competitions.codalab.org/competitions/19938

Author: W.M. Kouw
Date: 22-10-2018
"""
import os
import numpy as np
import pandas as pd
import pickle as pc
import dateutil.parser
from glob import glob
import json
import codecs

from nltk.tokenize.api import StringTokenizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

# Whether to embed words
embed = True

# Set font size
fS = 20

# Change to twitter data dir
os.chdir('/home/wmkouw/Dropbox/Projects/ucopenhagen/seq-rumour/data/RumEval2019')

# Get labels
with open('train-key.json') as f:
    train_key = json.load(f)

with open('dev-key.json') as f:
    dev_key = json.load(f)

label_keys = {**train_key['subtaskaenglish'], **dev_key['subtaskaenglish']}

# Get folder paths
twitter_path = 'twitter-english/'
rumours = os.listdir(twitter_path)

# Text array
rumour_id = []
tweet_id = []
thread_ix = []
reply_ix = []
texts = []
created_date = []
created_datetime = []
labels = []

# Loop over rumours
for r, rumour in enumerate(rumours):

    # Check threads for current rumour
    threads = os.listdir(twitter_path + rumour)

    # Loop over threads
    for t, thread in enumerate(threads):

        with open(twitter_path + rumour + '/' + thread + '/source-tweet/' + thread + '.json') as f:
            tweet = json.load(f)

            rumour_id.append(rumour)
            tweet_id.append(thread)
            thread_ix.append(t)
            reply_ix.append(0)
            texts.append(tweet['text'])
            created_date.append(dateutil.parser.parse(tweet['created_at']).date())
            created_datetime.append(dateutil.parser.parse(tweet['created_at']))
            labels.append(label_keys[thread])

        replies = os.listdir(twitter_path + rumour + '/' + thread + '/replies/')
        for r, reply in enumerate(replies):

            with open(twitter_path + rumour + '/' + thread + '/replies/' + reply) as f:
                tweet = json.load(f)

                rumour_id.append(rumour)
                tweet_id.append(reply[:-5])
                thread_ix.append(t)
                reply_ix.append(r + 1)
                texts.append(tweet['text'])
                created_date.append(dateutil.parser.parse(tweet['created_at']).date())
                created_datetime.append(dateutil.parser.parse(tweet['created_at']))
                labels.append(label_keys[reply[:-5]])

# Convert to dataframe
data = pd.DataFrame({'id': tweet_id,
                     'rumour': rumour_id,
                     'thread_ix': thread_ix,
                     'reply_ix': reply_ix,
                     'text': texts,
                     'date': created_date,
                     'datetime': created_datetime,
                     'label': labels})

# write frame to csv
data.to_csv('./RumEval19.csv', sep='`', encoding='utf-8')

if embed:

    # Change directory to word2vec model
    os.chdir('/home/wmkouw/Dropbox/Projects/ucopenhagen/seq-rumour/data/word2vec-twitter')

    #!! change 'xrange' in word2vecReader to 'range'
    exec(open("repl.py").read())

    # Start tokenizer
    tt = TweetTokenizer()

    # Check number of tweets
    num_tweets = len(data)

    # Loop over tweets
    wemb = np.zeros((num_tweets, 400))
    for n in range(num_tweets):

        # Tokenize tweet
        aa = tt.tokenize(data['text'][n])

        # Loop over words
        ct = 0
        for a in aa:

            try:
                # Extract embedding of word and add
                wemb[n, :] += model.__getitem__(a)
                ct += 1
            except:
                print('.', end='')

        # Average embeddings
        wemb[n, :] /= ct

    # Switch back to data dir
    os.chdir('/home/wmkouw/Dropbox/Projects/ucopenhagen/seq-rumour/data/RumEval2019')

    # Write embbeding array separately
    np.save('rumeval19.npy', wemb)

    # Add word embeddings to dataframe
    data = data.assign(embedding=wemb.tolist())

    # write frame to csv
    data.to_csv('./RumEval19_emb.csv', sep='\t', encoding='utf-8', index=False)
