import numpy as np
import seaborn as sns
import pandas as pd
import sys
import os
import numpy
import matplotlib.pyplot as plt
import warnings

numpy.set_printoptions(threshold=sys.maxsize)
numpy.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def read_table_with_join(path='/content/drive/MyDrive/Colab Notebooks/outputs/'):
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files]
    table = pd.DataFrame()
    for f in files:
        _table = pd.read_csv(f, index_col=[0])
        table = pd.concat([table, _table])
    new_names = {'batle': 'C-Batle',
                 'dragonnet': 'Drag.',
                 'bdragonnet': 'B-Drag.',
                 'cevae': 'Cevae',
                 'aipw': 'AIPW'}

    table['model_name'] = [new_names[item] for item in table['model_name']]
    table['source_size_p'] = table['source_size_p'] * 100
    table['source_size_p'] = [str(round(item)) + '%' for item in table['source_size_p']]
    ratios = {
        '20%': '(' + str(0.25) + ')',
        '40%': '(' + str(0.67) + ')',
        '60%': '(' + str(1.5) + ')',
        '80%': '(' + str(4) + ')'
    }
    table['source_size_p'] = [item + ratios[item] for item in table['source_size_p']]
    table_stats = table[['model_name', 'source_size_p', 'mae_naive', 'mae_aipw']]
    print(table_stats.groupby(['model_name', 'source_size_p']).mean())

    return table


def read_table(filename,
               path='/content/drive/MyDrive/Colab Notebooks/outputs/',
               quick_stats=False):
    path = os.path.join(path, filename + '.csv')
    table = pd.read_csv(path, index_col=[0])
    if quick_stats:
        try:
            table_stats = table[['model_name', 'source_size_p', 'mae_naive', 'mae_aipw']]
            print(table_stats.groupby(['model_name', 'source_size_p']).mean())
        except:
            table['mae_naive'] = table['tau'] - table['ate_naive_all']
            table['mae_aipw'] = table['tau'] - table['ate_aipw_all']
            table['mae_naive'] = np.abs(table['mae_naive'].values)
            table['mae_aipw'] = np.abs(table['mae_aipw'].values)
            # table = table[table['mae_naive']<2]
            table_stats = table[['model_name', 'source_size_p', 'mae_naive', 'mae_aipw']]
            print(table_stats.groupby(['model_name', 'source_size_p']).mean())
    return table


def set_colors(methods_order,
               our_method='C-Batle',
               two_colors=['#FF8C00', '#1e90ff']):
    # two_colors: first posision are baselines, second are our proposed method
    return [two_colors[1] if method == our_method else two_colors[0] for method in methods_order]


def single_barplot(table, metric_name, metric_name_ylabel, title,
                   save_plot=False, fontsize=15, font_scale=1.3,
                   methods_order=None, log_scale=False,
                   data_name=''
                   ):
    sns.set(font_scale=font_scale)
    if not methods_order:
        methods_order = pd.unique(table['model_name'].values)
    colors_order = set_colors(methods_order=methods_order)
    ax = sns.barplot(x='model_name', y=metric_name,
                     palette=sns.color_palette(colors_order),
                     dodge=False, data=table)  # hue='model_name'order=order,
    ax.set_xlabel("Method", fontsize=fontsize)
    if log_scale:
        ax.set_yscale("log")
        metric_name_ylabel = metric_name_ylabel + ' (log scale)'
    ax.set_ylabel(metric_name_ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    if save_plot:
        plt.savefig(data_name + '_single_plot.png', dpi=300, bbox_inches='tight')
    return ax


def set_plots(table, metric_name_y, metric_name_ylabel, title,
              group_name_x, group_name_xlabel,
              save_plot=False, fontsize=15, font_scale=1.3,
              figsize=(11.7, 8.27), axis_max=0.3,
              _h=['o', '\\', '+', ''], log_scale=False,
              methods_order=None, data_name='', ncol_legend=3):
    sns.set(rc={'figure.figsize': figsize})
    sns.set(font_scale=font_scale)
    methods_rep = pd.unique(table['source_size_p'].values)
    if not methods_order:
        methods_order = pd.unique(table['model_name'].values)

    colors_order = set_colors(methods_order=methods_order)
    ax = sns.barplot(x=group_name_x, y=metric_name_y,
                     palette=sns.color_palette(colors_order),
                     saturation=0.8, data=table,
                     hue='model_name')
    if log_scale:
        ax.set_yscale("log")
        metric_name_ylabel = metric_name_ylabel + ' (log scale)'
    ax.set_ylabel(metric_name_ylabel, fontsize=fontsize)
    ax.set_xlabel(group_name_xlabel, fontsize=fontsize)
    ax.axis(ymin=0, ymax=axis_max)  # 2.1

    # Define some hatches
    hatches = [np.repeat(h, len(methods_rep)) for h in _h]
    hatches = [item for sublist in hatches for item in sublist]
    # Loop over the bars
    for i, thisbar in enumerate(ax.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[i])
    ax.legend(ncol=ncol_legend, loc='upper right', fontsize=fontsize)
    ax.xaxis.get_label().set_fontsize(fontsize)
    if save_plot:
        plt.savefig(data_name + '_' + title + '.png', dpi=300, bbox_inches='tight')
    return ax


def plot_around_treat(table, seed, metric_name, labely_name, methods_order,
                      save_plot=False, fontsize=15, font_scale=1.3, figsize=(8, 5),
                      title='swarm', data_name=''):
    sns.set(rc={'figure.figsize': figsize})
    sns.set(font_scale=font_scale)
    taus = pd.unique(table['tau'])
    table = table[table['tau'] == taus[seed]]
    table['source_size_p'] = ['p=' + str(item) for item in table['source_size_p']]
    sns.set(font_scale=font_scale)
    colors_order = set_colors(methods_order=methods_order)
    ax = sns.swarmplot(x='model_name', y=metric_name,
                       data=table,
                       palette=sns.color_palette(colors_order),
                       size=8
                       )
    ax.set_xticklabels(ax.get_xticklabels())  # rotation=90
    ax.axhline(y=taus[seed], color='black', linestyle='-', linewidth=3)
    tau = round(taus[seed], 2)
    ax.set_ylabel(labely_name + '(τ=' + str(tau) + ')', fontsize=fontsize)
    ax.set_xlabel('Dataset Replication ' + str(seed), fontsize=fontsize)
    if save_plot:
        plt.savefig(data_name + '_' + title + str(seed) + '.png', dpi=300, bbox_inches='tight')
    return ax
