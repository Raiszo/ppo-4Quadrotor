import sys
import matplotlib
if sys.platform == 'darwin':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from os import path as path

"""
Using the plotter:
Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure
    data
    L test_EnvName_DateTime
      L params.json
      L  0
        L log.txt
      L  1
        L log.txt
       .
       .
       .
      L  9
        L log.txt
To plot learning curves from the experiment, averaged over all random
seeds, call
    python plot.py data/test_EnvName_DateTime --value AverageReturn
and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.
Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call
    python plot.py data/test1 data/test2
and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.
"""

def plot_data(data, plot_name, value="AverageReturn"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    # plt.ioff()
    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition")
    plt.legend(loc='best').draggable()
    plt.savefig(plot_name, dpi='figure', bbox_inches='tight')
    plt.show()


def get_datasets(fpath, condition=None):
    params_path = path.join(fpath, 'params.json')
    assert path.exists(params_path), "params.json must exist at the root of the experiment folder"

    with open(params_path) as f:
        params= json.load(f)
        
    exp_name = params['exp_name']

    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            log_path = path.join(root,'log.txt')
            experiment_data = pd.read_table(log_path)

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition or exp_name
                )

            datasets.append(experiment_data)
            unit += 1

    return datasets
            
def main():
    fpath = 'experiments/PPO-00_Pendulum-v0_24-10-2018_20-44-54'
    data = get_datasets(fpath)
    plot_data(data, path.join(fpath, 'plot4this.png'))



# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('logdir', nargs='*')
#     parser.add_argument('--legend', nargs='*')
#     parser.add_argument('--value', default='AverageReturn', nargs='*')
#     args = parser.parse_args()

#     use_legend = False
#     if args.legend is not None:
#         assert len(args.legend) == len(args.logdir), \
#             "Must give a legend title for each set of experiments."
#         use_legend = True
        
#     data = []
#     if use_legend:
#         for logdir, legend_title in zip(args.logdir, args.legend):
#             data += get_datasets(logdir, legend_title)
#     else:
#         for logdir in args.logdir:
#             data += get_datasets(logdir)

#     if isinstance(args.value, list):
#         values = args.value
#     else:
#         values = [args.value]
#     for value in values:
#         plot_data(data, path.join(logdir, 'plot4this.png'), value=value)

if __name__ == "__main__":
    main()
