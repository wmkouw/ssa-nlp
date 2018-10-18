"""
Read JSON files in PHEME tarball.

PHEME is an EU FP7 project aimed at analyzing rumours and "false news" on 
social media. One of the goals is to predict rumour veracity from language use.
https://www.pheme.eu/

Author: W.M. Kouw
Date: 15-10-2018
"""
import json
from glob import glob
from os import listdir, walk
from os.path import isdir
import pandas as pd
import numpy as np
import dateutil.parser
import tarfile

# Check if PHEME tarball has been downloaded
if not isdir("threads"):
    raise IOError("'threads' folder from PHEME not found. Please download and \
                    unpack the tarball in this directory, from \
                    https://ndownloader.figshare.com/files/4988998")

# Initialize lists
index = []
content = []
created_date = []
created_datetime = []
is_rumour = []
which_rumour = []
misinformation = []

# Set base path
threads_path = './threads/en/'

# Rumour-sets
rumours = listdir(threads_path)

# Counter
cnt = 0

# Loop over rumours
for rumour in rumours:

    # Current rumour path
    rumour_path = threads_path + rumour + '/'

    # Find all source tweets
    tweet_numbers = listdir(rumour_path)

    # Loop over tweets in current rumour
    for tweet_number in tweet_numbers:

        # Current tweet path
        tweet_path = rumour_path + tweet_number + '/'

        # Read twitter
        with open(tweet_path + 'source-tweets/' + tweet_number + '.json') as f:
            data = json.load(f)

        with open(tweet_path + 'annotation.json') as f:
            anno = json.load(f)

        # Store text, time and rumour annotation in numpy array
        index.append(cnt)
        content.append(data['text'])
        created_date.append(dateutil.parser.parse(data['created_at']).date())
        created_datetime.append(dateutil.parser.parse(data['created_at']))
        is_rumour.append(anno['is_rumour'])
        which_rumour.append(rumour)
        misinformation.append(anno['misinformation'])

        # Increment counter
        cnt += 1

# Convert to dataframe
tweets = pd.DataFrame({'content': content,
                       'created_date': created_date,
                       'created_datetime': created_datetime,    
                       'is_rumour': is_rumour,
                       'which_rumour': which_rumour,
                       'misinformation': misinformation})

# write frame to csv
tweets.to_csv('./PHEME.csv', sep='\t', encoding='utf-8', index=False)
