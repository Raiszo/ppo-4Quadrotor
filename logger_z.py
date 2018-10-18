import json

"""

Some simple logging functionality, inspired by rllab's logging.
Assumes that each diagnostic gets logged each iteration

Call logz.configure_output_dir() to start logging to a 
tab-separated-values file (some_folder_name/log.txt)

To load the learning curves, you can do, for example

A = np.genfromtxt('/tmp/expt_1468984536/log.txt',delimiter='\t',dtype=None, names=True)
A['EpRewMean']

"""

import os.path as osp, shutil, time, atexit, os, subprocess
import pickle
# import tensorflow as tf

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

class Logger:
    def __init__(self, log_dir):
        assert not osp.exists(G.output_dir), "Log dir %s already exists! Delete it first or use a different dir"%G.output_dir

        self.output_dir = log_dir
        os.makedirs(G.output_dir)
        
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

        
        self.output_file = open(osp.join(G.output_dir, "log.txt"), 'w')
        atexit.register(G.output_file.close)
        print(colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        """
        if self.first_row:
            G.log_headers.append(key)
        else:
            assert key in G.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
            assert key not in G.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
            G.log_current_row[key] = val
