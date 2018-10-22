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

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class Logger:
    def __init__(self, output_dir):
        # print(output_dir)
        assert not osp.exists(output_dir), "Log dir %s already exists! Delete it first or use a different dir"%output_dir

        self.output_dir = output_dir
        os.makedirs(output_dir)
        
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

        
        self.output_file = open(osp.join(output_dir, "log.txt"), 'w')
        atexit.register(self.output_file.close)
        print(colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))

    @staticmethod
    def save_params(exp_dir, params):
        with open(osp.join(exp_dir, "params.json"), 'w') as out:
            out.write(json.dumps(params, separators=(',\n','\t:\t'), sort_keys=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    # def dump_tabular(print_console):
    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            if hasattr(val, "__float__"): valstr = "%8.3g"%val
            else: valstr = val
            vals.append(val)

        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers))
                self.output_file.write("\n")
            self.output_file.write("\t".join(map(str, vals)))
            self.output_file.write("\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False
