import inspect
from pathlib import Path
from typing import Callable
from tbparse import SummaryReader
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


def plot_runs(runs, title, winsize=100, ax=None):
    train_rewards_dfs = []
    test_rewards_dfs = []
    if inspect.isgenerator(runs):
        rundirs = runs
    else:
        rundirs = runs()

    for run_dir in rundirs:
        print(run_dir)
        reader = SummaryReader(run_dir)
        df = reader.scalars
        train_rewards_df = df[df['tag'] == 'train/reward'][['step', 'value']].copy()
        train_rewards_df.set_index('step', inplace=True)
        train_rewards_dfs.append(train_rewards_df.copy())

        test_rewards_df = df[df['tag'] == 'test/reward'][['step', 'value']].copy()
        test_rewards_df.set_index('step', inplace=True)
        test_rewards_dfs.append(test_rewards_df.copy())

    train_rewards_dfs = pd.concat(train_rewards_dfs, axis=0)
    train_rewards_dfs.sort_index(inplace=True)
    train_rewards_dfs['mean'] = train_rewards_dfs['value'].rolling(window=winsize).mean()
    train_rewards_dfs['std'] = train_rewards_dfs['value'].rolling(window=winsize).std()
    train_rewards_dfs.dropna(inplace=True)

    test_rewards_dfs = pd.concat(test_rewards_dfs, axis=0)
    test_rewards_dfs.sort_index(inplace=True)
    test_rewards_dfs['mean'] = test_rewards_dfs['value'].rolling(window=winsize).mean()
    test_rewards_dfs['std'] = test_rewards_dfs['value'].rolling(window=winsize).std()
    test_rewards_dfs.dropna(inplace=True)

    if ax is None:
        ax = plt.gca()

    for dat, c, label in [(train_rewards_dfs, 'b', 'Train'), (test_rewards_dfs, 'r', 'Test')]:
        ax.plot(dat.index, dat['mean'], color=c)
        ax.fill_between(x=dat.index, y1=dat['mean'] - dat['std'], y2=dat['mean'] + dat['std'],
                        color=c, alpha=0.3, label=label)

    ax.legend()
    ax.set_title(title)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: '%1.1f' % (x * 1e-6)))
    ax.set_ylabel('Reward')
    ax.set_xlabel('Steps/Million')


def extract_data(runs, winsize=100):
    train_rewards_dfs = []
    test_rewards_dfs = []
    if inspect.isgenerator(runs):
        rundirs = runs
    else:
        rundirs = runs()

    for run_dir in rundirs:
        print(run_dir)
        reader = SummaryReader(run_dir)
        df = reader.scalars
        train_rewards_df = df[df['tag'] == 'train/reward'][['step', 'value']].copy()
        train_rewards_df.set_index('step', inplace=True)
        train_rewards_dfs.append(train_rewards_df.copy())

        test_rewards_df = df[df['tag'] == 'test/reward'][['step', 'value']].copy()
        test_rewards_df.set_index('step', inplace=True)
        test_rewards_dfs.append(test_rewards_df.copy())

    train_rewards_dfs = pd.concat(train_rewards_dfs, axis=0)
    train_rewards_dfs.sort_index(inplace=True)
    train_rewards_dfs['mean'] = train_rewards_dfs['value'].rolling(window=winsize).mean()
    train_rewards_dfs['std'] = train_rewards_dfs['value'].rolling(window=winsize).std()
    train_rewards_dfs.dropna(inplace=True)

    test_rewards_dfs = pd.concat(test_rewards_dfs, axis=0)
    test_rewards_dfs.sort_index(inplace=True)
    test_rewards_dfs['mean'] = test_rewards_dfs['value'].rolling(window=winsize).mean()
    test_rewards_dfs['std'] = test_rewards_dfs['value'].rolling(window=winsize).std()
    test_rewards_dfs.dropna(inplace=True)

    return train_rewards_dfs, test_rewards_dfs

# def normal():
#     for run_dir in res_dir.glob('tasks.MGFR-9x9-RARG-GI-Q-*'):
#         if 'HER' not in run_dir.name:
#             yield run_dir


# def HER():
#     for run_dir in res_dir.glob('tasks.MGFR-9x9-RARG-GI-Q-HER-*'):
#         if 'customized' not in run_dir.name:
#             yield run_dir


# def HER_customized():
#     for run_dir in res_dir.glob('tasks.MGFR-9x9-RARG-GI-Q-HER-*'):
#         if 'customized' in run_dir.name:
#             yield run_dir

# fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

# plot_runs(normal, 'Normal Replay', ax=axes[0])
# # plot_runs(HER, 'HER (future strategy) Replay', ax=axes[1])
# plot_runs(HER_customized, 'HER (customized strategy) Replay', ax=axes[1])

# plt.tight_layout()
# plt.savefig('figs/tf.png')

def universal_run_filter(common_prefix: str, specific_condition: Callable, result_dir: Path):
    for run_dir in result_dir.glob(common_prefix):
        if specific_condition(run_dir.name) is True:
            yield run_dir


res_dir = Path('results')
comm_prefix = 'DQN-tasks.TVMGFR-19x19-RARG-GI-NewMethod*'

def off_cond(x):
    if 'subgoal-off' not in x:
        return False
    if '2023' not in x:
        return False
    return int(x.split('-')[-1]) >= 20230407095812 and int(x.split('-')[-1]) <= 20230407132416


def on_cond(x):
    if 'subgoal-on' not in x:
        return False
    if '2023' not in x:
        return False
    return int(x.split('-')[-1]) >= 20230407033015 and int(x.split('-')[-1]) <= 20230407091606


subgoal_off = universal_run_filter(comm_prefix, off_cond, res_dir)
subgoal_on = universal_run_filter(comm_prefix, on_cond, res_dir)


df_train_off, df_test_off = extract_data(subgoal_off)
df_train_on, df_test_on = extract_data(subgoal_on)
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)


def plot_avg_performance(ax, c, label, dat, alpha=0.3):
    ax.plot(dat.index, dat['mean'], color=c)
    ax.fill_between(x=dat.index, y1=dat['mean'] - dat['std'], y2=dat['mean'] + dat['std'],
                    color=c, alpha=alpha, label=label)


plot_avg_performance(axes[0], 'blue', 'Subgoal OFF', df_train_off)
plot_avg_performance(axes[0], 'red', 'Subgoal ON', df_train_on)

plot_avg_performance(axes[1], 'blue', 'Subgoal OFF', df_test_off)
plot_avg_performance(axes[1], 'red', 'Subgoal ON', df_test_on)
axes[0].legend()
axes[1].legend()


plt.tight_layout()
plt.savefig('figs/subgoal-on-off.png')
