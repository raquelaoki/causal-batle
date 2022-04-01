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
               our_method='batle',
               two_colors=['#FF8C00', '#1e90ff']):
    # two_colors: first posision are baselines, second are our proposed method
    return [two_colors[1] if method == our_method else two_colors[0] for method in methods_order]


def single_barplot(table, metric_name, metric_name_ylabel, title,
                   save_plot=False, fontsize=15, font_scale=1.3):
    sns.set(font_scale=font_scale)
    methods_order = pd.unique(table['model_name'].values)
    colors_order = set_colors(methods_order=methods_order)
    ax = sns.barplot(x='model_name', y=metric_name,
                     palette=sns.color_palette(colors_order),
                     dodge=False, data=table)  # hue='model_name'order=order,
    ax.set_xlabel("Method", fontsize=fontsize)
    ax.set_ylabel(metric_name_ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    if save_plot:
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
    return ax


def set_plots(table, metric_name_y, metric_name_ylabel, title,
              group_name_x, group_name_xlabel,
              save_plot=False, fontsize=15, font_scale=1.3,
              figsize=(11.7, 8.27), axis_max=0.3,
              _h = ['o', '\\', '+', '']):
    sns.set(rc={'figure.figsize': figsize})
    sns.set(font_scale=font_scale)
    methods_order = pd.unique(table['model_name'].values)
    colors_order = set_colors(methods_order=methods_order)
    ax = sns.barplot(x=group_name_x, y=metric_name_y,
                     palette=sns.color_palette(colors_order),
                     saturation=0.8, data=table, hue='model_name')
    ax.set_xlabel(group_name_xlabel, fontsize=fontsize)
    ax.set_ylabel(metric_name_ylabel, fontsize=fontsize)
    ax.axis(ymin=0, ymax=axis_max)  # 2.1
    # Define some hatches
    hatches = [np.repeat(h, len(methods_order)) for h in _h]
    hatches = [item for sublist in hatches for item in sublist]
    # Loop over the bars
    for i, thisbar in enumerate(ax.patches):
        # Set a different hatch for each bar
        thisbar.set_hatch(hatches[i])
    ax.legend(ncol=2, loc='upper right', fontsize=fontsize)
    ax.xaxis.get_label().set_fontsize(fontsize)
    if save_plot:
        plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
    return ax
