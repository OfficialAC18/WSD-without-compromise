#Description: Main file for running the program
# 1. Set random state for the run, then generate a random seed for the model
#
#
#
#
#
#
#
#

import os
import logging
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, required=True, help='Directory to model checkpoints \
                                                            (models are trained if directory is empty)')
parser.add_argument('--model_configs', type=str, required=True, help='Path to model configurations')
parser.add_argument('--data_dir', type=str, default='datasets', help='Directory to data')
parser.add_argument('--pipeline_seed', type=int, default=42, help='Seed for the pipeline')
parser.add_argunment('--eval_seed', type=int, default=42, help='Seed for running evaluation')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing model checkpoints')

def main(args):
    #For reproducibility
    np.random.seed(args.pipeline_seed)

    if not os.path.exists(args.model_dir) or args.overwrite:
        os.makedirs(args.model_dir, exist_ok=True)
        logging.warning(f'Training Models, saving to {args.model_dir}')
    else:
        logging.warning(f'Loading models from {args.model_dir}')
    


    return




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



