import argparse
import multiprocessing
import os
import pickle
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

## Importing the models
from pygorl.cogpolicy import ACLPolicyGradient, AdvLPolicyGradient, VLPolicyGradient
from pygorl.cogq import (
    FQLearning,
    HetFQLearning,
    HetOSFQLearning,
    HetOSQLearning,
    HetQLearning,
    HetSOSFQLearning,
    OSFQLearning,
    OSQLearning,
    QLearning,
    SOSFQLearning,
)
from pygorl.rdp_client import unlock_and_unzip_file
from tqdm import tqdm

# create a parser object
parser = argparse.ArgumentParser(description="Fit a model to a dataset")
# add arguments with default values
parser.add_argument(
    "--model",
    default="QL",
    type=str,
    help="Model to fit (valid options: (Het)QL, (Het)FQL, (Het)OSQL, (Het)OSFQL, (Het)SOSFQL), VLP, ACLP, AdvLP",
)
parser.add_argument("--algorithm", default="de", type=str, help="Algorithm to use (valid options: de, shgo, minimize)")
parser.add_argument("--n_modules", default=2, type=int, help="Number of modules for heterogeneous models")
parser.add_argument(
    "--mix_rule",
    default="weighted",
    type=str,
    help="Mixing function for heterogeneous models (valid options: weighted, max)",
)
parser.add_argument(
    "--q_type", default="q", type=str, help="Type of Q-values to use for AC policy (valid options: q, fq, osfq)"
)
parser.add_argument("--k", default=2, type=int, help="Number of folds for cross-validation")
parser.add_argument(
    "--randomize",
    default=-1,
    type=int,
    help="Whether to randomize the data before fitting (valid options: -1 for no, positive integer for seed)",
)
parser.add_argument("--n_jobs", default=2, type=int, help="Number of cores to use, -1 for all cores")
parser.add_argument("--lambda_reg", default=0, type=float, help="Regularization parameter")
parser.add_argument(
    "--data",
    default="data/dmData_06-07-2023.ezip",
    type=str,
    help="Path to encrypted data file (if multifile, give the first file)",
)
parser.add_argument("--output", default="processed_data/dmData_06-07-2023/", type=str, help="Path to output directory")
parser.add_argument(
    "--qc", default="full", type=str, help="Whether to perform quality control (valid options: minimal, full, none)"
)
parser.add_argument(
    "--filterdate", default="none", type=str, help="Whether to filter by date (valid options: none, yyyy-mm-dd)"
)

# parse the arguments from standard input
args = parser.parse_args()

start_string = """
PyGORL: Python based fitting of Globally Optimized Reinforcement Learning algorithms
====================================================================================
Author: Rishika Mohanta, Turner Lab, Janelia Research Campus, Ashburn VA
"""
print(start_string)


# check for valid model
valid_models = ["QL", "FQL", "OSQL", "OSFQL", "SOSFQL"]
valid_models += ["Het" + m for m in valid_models]
valid_models += ["VLP", "ACLP", "AdvLP"]
assert args.model in valid_models, "Invalid model"

if args.model == "ACLP":
    assert args.q_type in ["q", "fq", "osfq"], "Invalid Q-value type"

# check for valid algorithm
assert args.algorithm in ["de", "shgo", "minimize"], "Invalid algorithm"

# setup algorithm parameter defaults
algo_params = {}
if args.algorithm == "de":
    algo_params["popsize"] = 100
    algo_params["tol"] = 1e-3
    algo_params["disp"] = True
if args.algorithm == "shgo":
    algo_params["options"] = {"disp": True, "maxiter": 300}
    algo_params["iters"] = 1
if args.algorithm == "minimize":
    algo_params["maxiter"] = 1000
    algo_params["tol"] = 1e-3
    algo_params["disp"] = True
    algo_params["randomize"] = True
    algo_params["n_restarts"] = 20

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
if not (args.data.endswith(".ezip") or args.data.endswith(".ezip.000")) or not os.path.exists(args.data):
    # append to current working directory and display
    print("Data path: ", os.getcwd() + "/" + args.data)
    print("Warning: Invalid data path, provide path to encrypted data file")
    exit(1)

# check if a directory with the same name as the data file exists
if os.path.exists(args.data.split(".ezip")[0] + "/"):
    print("Warning: Directory with same name as data file exists, overwriting")
    shutil.rmtree(args.data.split(".ezip")[0] + "/")

# decrypt and unzip the data file
if len(args.data.split(".")) == 2:
    unlock_and_unzip_file(args.data)
elif len(args.data.split(".")) == 3:
    unlock_and_unzip_file(args.data, multifile=True)
else:
    raise Exception("Invalid data path")

data_path = args.data.split(".ezip")[0] + "/"

# check for valid output path creating all directories if necessary
if not os.path.exists(args.output):
    print("Warning: Invalid output path, creating directory")
    os.makedirs(args.output)

# check for valid quality control option
assert args.qc in ["minimal", "full", "none"], "Invalid quality control option"
if args.filterdate != "none":
    try:
        last_date = args.filterdate + " 00:00:00"
        args.filterdate = pd.to_datetime(args.filterdate, format="%Y-%m-%d %H:%M:%S")
    except:
        print("Warning: Invalid date format, skipping date filter")
        last_date = "none"
else:
    last_date = "none"

# Importing the dataset
print("Loading data from folder: ", data_path)
choices_full = np.loadtxt(data_path + "choices.csv", delimiter=",")
rewards_full = np.loadtxt(data_path + "rewards.csv", delimiter=",")
assert choices_full.shape == rewards_full.shape, "Choices and rewards are not the same shape"
print(
    "Data loaded successfully with N = {} flies and {} maximum trials".format(
        choices_full.shape[0], choices_full.shape[1]
    )
)

N = choices_full.shape[0]  # number of flies

# get meta and qc
meta = pd.read_csv(data_path + "metadata.csv")
qc = np.loadtxt(data_path + "quality_control.csv", delimiter=",").astype(bool)

# remove control flies
choices_full = choices_full[meta["Fly Experiment"] != "control.csv"]
rewards_full = rewards_full[meta["Fly Experiment"] != "control.csv"]
qc = qc[meta["Fly Experiment"] != "control.csv"]
meta = meta[meta["Fly Experiment"] != "control.csv"]
choices_full = choices_full[meta["Fly Experiment"] != "control_reciprocal.csv"]
rewards_full = rewards_full[meta["Fly Experiment"] != "control_reciprocal.csv"]
qc = qc[meta["Fly Experiment"] != "control_reciprocal.csv"]
meta = meta[meta["Fly Experiment"] != "control_reciprocal.csv"]
meta.reset_index(drop=True, inplace=True)

# Perform quality control
if args.qc == "minimal":
    choices_full = choices_full[qc]
    rewards_full = rewards_full[qc]
if args.qc == "full":
    meta = meta[qc]
    meta = meta[meta["Experiment Start Time"] < last_date].groupby("Fly Experiment").head(3)
    choices_full = choices_full[meta.index]
    rewards_full = rewards_full[meta.index]
print(
    "{}/{} ({}) flies passed quality control".format(
        choices_full.shape[0], N, "{:0.2f}".format(choices_full.shape[0] / N * 100)
    )
)

if args.randomize > 0:
    print("Randomizing data with seed {}".format(args.randomize))
    np.random.seed(args.randomize)
    order = np.random.permutation(choices_full.shape[0])
    choices_full = choices_full[order]
    rewards_full = rewards_full[order]

# Set up the model and algorithm
if "QL" in args.model:
    if "Het" in args.model:
        model = eval(args.model + "earning(N_modules=args.n_modules,mix_rule=args.mix_rule)")
    else:
        model = eval(args.model + "earning()")
else:
    if args.model == "ACLP":
        model = eval(args.model + "olicyGradient(q_type=args.q_type)")
    else:
        model = eval(args.model + "olicyGradient()")

model_name = args.model
algorithm = args.algorithm
K = args.k  # number of folds
n_jobs = args.n_jobs  # number of cores to use
lambda_reg = args.lambda_reg  # regularization parameter

# Print out the model and algorithm
start_str = "Initializing fit of {} model using {} algorithm with {} cores".format(model_name, algorithm, n_jobs)
print(start_str)
print("".join(["="] * len(start_str)))
print("Parameters to be estimated: ", ", ".join(model.param_props()["names"]))


# Set up the initial parameters and bounds
params_init = model.param_props()["suggested_init"]
params_bounds = model.param_props()["suggested_bounds"]
print("Initial parameters and bounds loaded successfully")

if K > 1:
    print("Starting fitting process with {} folds".format(K))
    results = Parallel(n_jobs=n_jobs)(
        delayed(model.fit_every_nth)(
            start,
            K,
            choices=choices_full,
            rewards=rewards_full,
            params_init=params_init,
            lambda_reg=lambda_reg,
            bounds=params_bounds,
            algo=algorithm,
            **algo_params,
        )
        for start in tqdm(range(K))
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
        **algo_params,
    )

# save a single file
dt = datetime.now().strftime("%Y%m%d")

if args.model == "ACLP":
    model_name = args.model + "_" + args.q_type.upper()
if "Het" in args.model:
    model_name = "Het" + args.n_modules + model_name[3:] + "_" + args.mix_rule[0]

with open(args.output + f"{model_name}_{K}cv_{algorithm}_{dt}.pkl", "wb") as f:
    pickle.dump(results, f)

print("Warning: Deleting decrypted data")
if os.path.exists(data_path):
    shutil.rmtree(data_path)

