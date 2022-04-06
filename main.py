"""
Written by Raquel Aoki
"""
import logging
import os
import pandas as pd
import sys
import yaml
import numpy as np
# Local Imports
from utils import read_config_names, repeat_experiment
import helper_parameters as hp

def main(paths_args):
    """ Main function of the project.
    It loads config settings, dataset, run all the methods, save output.
    Args:
        paths: path for the config files.
    """

    # Find the path+name of all config files in a given  folder
    configs = read_config_names(paths_args['config'])

    #if paths_args['load_previous_table']:
    #    table = pd.read_csv(paths_args['load_previous_table_path'], index_col=[0])
    #else:
    #    table = pd.DataFrame()

    # Run models for all config files
    for config in configs:
        table = pd.DataFrame()
        params = hp.parameter_loader(config_path=config)
        output_path = paths_args['drive'] + 'table_' + params['config_name']
        table = repeat_experiment(params, table, use_range_source_p=paths_args['use_range_source_p'],
                                  save=paths_args['save'], output_save=output_path,
                                  previous=paths_args['previous'])
        table['mae_naive'] = table['tau'] - table['ate_naive_all']
        table['mae_aipw'] = table['tau'] - table['ate_aipw_all']
        table['mae_naive'] = np.abs(table['mae_naive'].values)
        table['mae_aipw'] = np.abs(table['mae_aipw'].values)
        if paths_args['save']:
            table.to_csv(output_path + '.csv', sep=',')

    return table


if __name__ == "__main__":
    main(config_path=sys.argv[1])