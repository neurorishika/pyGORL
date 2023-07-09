import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from models import *
import pickle
import os
import multiprocessing
from datetime import datetime

# import package to manage command line arguments
import argparse

# create a parser object
parser = argparse.ArgumentParser(description='Fit a model to a dataset')
# add arguments with default values
parser.add_argument('--model', default='QL', type=str, 
                    help='Model to fit (valid options: (Het)QL, (Het)FQL, (Het)OSQL, (Het)OSFQL, (Het)SOSFQL)')
parser.add_argument('--algorithm', default='de', type=str,
                    help='Algorithm to use (valid options: de, shgo, minimize)')
parser.add_argument('--n_modules', default=2, type=int,
                    help='Number of modules for heterogeneous models')
parser.add_argument('--k', default=2, type=int,
                    help='Number of folds for cross-validation')
parser.add_argument('--n_jobs', default=2, type=int,
                    help='Number of cores to use, -1 for all cores')
parser.add_argument('--lambda_reg', default=0, type=float,
                    help='Regularization parameter')
parser.add_argument('--data', default='data/deepmind/dmData_06-07-2023/', type=str,
                    help='Path to data files')
parser.add_argument('--output', default='fitted_models/deepmind/', type=str,
                    help='Path to output directory')
parser.add_argument('--qc', default='full', type=str,
                    help='Whether to perform quality control (valid options: minimal, full, none)')

# parse the arguments from standard input
args = parser.parse_args()

start_string = """
Cognitive Modeling Framework in Python
======================================
Author: Rishika Mohanta 
Turner Lab, Janelia Research Campus
"""
print(start_string)

# check for valid model
valid_models = ['QL', 'FQL', 'OSQL', 'OSFQL', 'SOSFQL']
valid_models += ['Het'+m for m in valid_models]
assert args.model in valid_models, "Invalid model"

# check for valid algorithm
assert args.algorithm in ['de', 'shgo', 'minimize'], "Invalid algorithm"

# setup algorithm parameter defaults
algo_params = {}
if args.algorithm == 'de':
    algo_params['popsize'] = 100
    algo_params['tol'] = 1e-3
    algo_params['disp'] = True
if args.algorithm == 'shgo':
    algo_params['options'] = {'disp':True,'maxiter':300}
    algo_params['iters'] = 1
if args.algorithm == 'minimize':
    algo_params['maxiter'] = 1000
    algo_params['tol'] = 1e-3
    algo_params['disp'] = True
    algo_params['randomize'] = True
    algo_params['n_restarts'] = 20

# check for valid number of folds
if args.k < 2:
    print("Warning: Invalid number of folds requested, skipping cross-validation")
    args.k = 1

# check for valid number of cores
if args.n_jobs == -1:
    args.n_jobs = multiprocessing.cpu_count()
elif args.n_jobs > multiprocessing.cpu_count():
    print("Warning: More cores requested than available, using all cores")
    args.n_jobs = multiprocessing.cpu_count()
elif args.n_jobs < 1:
    print("Warning: Invalid number of cores requested, using 1 core")
    args.n_jobs = 1

# check for valid regularization parameter
assert args.lambda_reg >= 0, "Invalid regularization parameter"

# check for valid data path
assert os.path.exists(args.data), "Invalid data path"

# check for valid output path
if not os.path.exists(args.output):
    print("Warning: Invalid output path, creating directory")
    os.mkdir(args.output)

# check for valid quality control option
assert args.qc in ['minimal', 'full', 'none'], "Invalid quality control option"

# Importing the dataset
print("Loading data from folder: ", args.data)
choices_full = np.loadtxt(args.data+'choices.csv', delimiter=',')
rewards_full = np.loadtxt(args.data+'rewards.csv', delimiter=',')
assert choices_full.shape == rewards_full.shape, "Choices and rewards are not the same shape"
print("Data loaded successfully with N = {} flies and {} maximum trials".format(choices_full.shape[0], choices_full.shape[1]))

N = choices_full.shape[0] # number of flies

# Perform quality control
if args.qc == 'minimal':
    qc = np.loadtxt(args.data+'quality_control.csv', delimiter=',').astype(bool)
    choices_full = choices_full[qc]
    rewards_full = rewards_full[qc]
if args.qc == 'full':
    qc = np.loadtxt(args.data+'quality_control.csv', delimiter=',').astype(bool)
    meta = pd.read_csv(args.data+'metadata.csv')
    meta = meta[qc].groupby('Fly Experiment').head(3)
    choices_full = choices_full[meta.index]
    rewards_full = rewards_full[meta.index]
print("{}/{} ({}) flies passed quality control".format(choices_full.shape[0], N, "{:0.2f}".format(choices_full.shape[0]/N*100)))

# Set up the model and algorithm
if 'Het' in args.model:
    model = eval(args.model+"earning(args.n_modules)")
else:
    model = eval(args.model+"earning()")
model_name = args.model
algorithm = args.algorithm
K = args.k # number of folds
n_jobs = args.n_jobs # number of cores to use
lambda_reg = args.lambda_reg # regularization parameter

# Print out the model and algorithm
start_str = "Initializing fit of {} model using {} algorithm with {} cores".format(model_name, algorithm, n_jobs)
print(start_str)
print("".join(["="]*len(start_str)))
print("Parameters to be estimated: ", ", ".join(model.param_props()['names']))

# Set up the initial parameters and bounds
params_init = model.param_props()['suggested_init']
params_bounds = model.param_props()['suggested_bounds']
print("Initial parameters and bounds loaded successfully")

if K > 1:
    print("Starting fitting process with {} folds".format(K))
    results = Parallel(n_jobs=n_jobs)(
        delayed(model.fit_every_nth)(
            start, K,
            choices=choices_full,
            rewards=rewards_full,
            params_init=params_init,
            lambda_reg=lambda_reg,
            bounds=params_bounds,
            algo=algorithm,   
            **algo_params
            ) for start in tqdm(range(K))
        )
else:
    print("Starting fitting process no cross-validation")
    results = model.fit_all(
        choices=choices_full,
        rewards=rewards_full,
        params_init=params_init,
        lambda_reg=lambda_reg,
        bounds=params_bounds,
        algo=algorithm,   
        **algo_params
        )

# save a single file
dt = datetime.now().strftime('%Y%m%d')
with open(args.output+f'{model_name}_{K}cv_{algorithm}_{dt}.pkl','wb') as f:
    pickle.dump(results,f)







