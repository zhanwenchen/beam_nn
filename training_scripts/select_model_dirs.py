#!/usr/bin/env python

import os
import pandas as pd

if __name__ == "__main__":

    # get parent directory
    parent = os.getcwd().split('/')[-1]

    files = os.listdir('.')
    dirs = []
    for f in files:
        if os.path.isdir(f):
            if f[0:2] == 'k_':
                dirs = dirs + [f]
    # sort dirs
    dirs = sorted(dirs)

    # create dataframe with directories to models for each frequency
    model_dirs = pd.DataFrame(dirs)
    for i, f in enumerate(dirs):
        model_dirs.iloc[i] = os.path.join('..', '..', '..', f) 
    
    # write to csv
    model_dirs.to_csv('model_dirs.txt', sep=',', header=None)
