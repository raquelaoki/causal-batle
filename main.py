"""
Code purpose: Illustrate a ML file structure
Written by Raquel Aoki
Date: September 2021
This is a new test
For an example using absl.flags, visit https://abseil.io/docs/python/guides/flags
"""

import os
import pandas as pd
import sys
import utils
import yaml
from helper_data import make_dataset
import logging


def main(config_path):
    """ Main function of the project.
    It loads config settings, dataset, run all the methods, save output.
    Args:
        config_path: path for the config files.
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")

    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logging.info('Started')

    # Load the config file. All comments should end with a .
    with open(config_path) as f:
        config = yaml.safe_load(f)
    params = config["parameters"]

    # Load dataset.
    data_s, data_t = make_dataset(params)

    # Run methods.
    output = utils.run_methdos(data_s, data_t, params)

    # Save results.
    if params['save_output']:
        file_path = params['path_output'] + str(params['config']) + '.csv'
        output.to_csv(file_path)
    else:
        print(output)
    logging.info('Finished')
    return


if __name__ == "__main__":
    main(config_path=sys.argv[1])