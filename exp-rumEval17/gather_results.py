#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize results from experiments.

Author: W.M. Kouw
Date: 15-10-2018
"""
import numpy as np
import pandas as pd
import pickle as pc
import json

import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('results/results_rumEval17.csv')

# Find rumours
rumours = np.unique(results['rumours'])

# Initialize figure
fig, ax = plt.subplots(ncols=len(rumours), figsize=(20, 6), sharey=True)

# Plot figure
for r, rumour in enumerate(rumours):

    # Bar plot performance, for each day
    results[results['rumours'] == rumour].plot(ax=ax[r],
                                               kind='bar',
                                               x='days',
                                               y=['err_n', 'err_a'],
                                               rot=0)

    # Set axes properties
    ax[r].set_title(rumour)
    ax[r].set_ylim([0, 1])
    if r == 0:
        ax[r].set_ylabel('error')

plt.show()
