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
  #two_colors: first posision are baselines, second are our proposed method
  return [two_colors[1] if method==our_method else two_colors[0] for method in methods_order]
