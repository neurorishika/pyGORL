import argparse
import gzip
import multiprocessing
import os
import pickle
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

## Importing the data protection module
from pygorl.rdp_client import unlock_and_unzip_file
from tqdm import tqdm


# setup the neuronal model
class MushroomBody:
    # initialize the mushroom body
    def __init__(self, mu_inh=0.1, fr=0.9, lr=0.5, up_dr=5.0, fb_syn=0.1, fb_trans=0.1, fb_up=1.0, pbn_asym=0.5):
        self.fr = fr  # forgetting rate
        self.lr = lr  # learning rate
        self.eps = 1e-3  # small number to avoid division by zero
        self.w_KC_pMBON = np.array([1.0, 1.0])  # weights from KC to MBON (appetitive)
        self.w_KC_nMBON = np.array([1.0, 1.0])  # weights from KC to MBON (aversive)
        # mutual inhibition between MBONs (Felsenberg et al., 2018)
        self.w_nMBON_pMBON = -1.0 * mu_inh  # weight from aversive MBON to appetitive MBON
        self.w_pMBON_nMBON = -1.0 * mu_inh  # weight from appetitive MBON to aversive MBON
        # MBON to DAN feedback
        self.w_pMBON_pDANs = (
            -1.0 * fb_syn
        )  # weight from appetitive MBON to reward DANs (inhibitory to subtract reward expectation)
        self.w_nMBON_nDANs = (
            -1.0 * fb_syn
        )  # weight from aversive MBON to punishment DANs (inhibitory to subtract punishment expectation)
        self.w_pMBON_nDANs = (
            1.0 * fb_trans
        )  # weight from appetitive MBON to punishment DANs (excitatory to add reward expectation)
        self.w_nMBON_pDANs = (
            1.0 * fb_trans
        )  # weight from aversive MBON to reward DANs (excitatory to add punishment expectation)
        # Upwind Neuron inputs
        self.w_pMBON_U = (
            1.0 * up_dr * pbn_asym
        )  # weight from appetitive MBON to upwind neuron (appetitive means upwind will be activated)
        self.w_nMBON_U = (
            -1.0 * up_dr
        )  # weight from aversive MBON to upwind neuron (aversive means upwind will be inhibited)
        # Upwind Neuron feedback to Dopamine Neurons
        self.w_U_pDANs = 1.0 * fb_up  # weight from upwind neuron to reward DANs
        self.w_U_nDANs = 0.0  # weight from upwind neuron to punishment DANs
        # Activation function
        # self.activation = lambda x: (1 / self.eps if x > 1 / self.eps else x) if x > 0 else 0  # ReLU
        self.activation = lambda x: np.clip(x, 0, 1 / self.eps)  # bounded ReLU

    # get the upwind drive for each odor without causing plasticity
    def upwind_drive(self):
        """
        A function to calculate the upwind drive for each odor without causing plasticity
        
        Parameters:
        -----------
        time_since_last_trial: float
            time since last trial in seconds (used for homeostatic plasticity, for now we set it to 1 arbitrary trial length)
        """
        drives = []
        for KC_activation in [np.array([1, 0]), np.array([0, 1])]:

            # Step 1: NO HOMEOSTATIC PLASTICITY
            w_KC_pMBON_ = self.w_KC_pMBON #+ (1 - self.w_KC_pMBON) * (1 - np.exp(-self.fr))
            w_KC_nMBON_ = self.w_KC_nMBON #+ (1 - self.w_KC_nMBON) * (1 - np.exp(-self.fr))

            # Step 2: calculate the MBON activations
            MBON_activation = np.array(
                [
                    self.activation(np.dot(w_KC_pMBON_, KC_activation)),
                    self.activation(np.dot(w_KC_nMBON_, KC_activation)),
                ]
            )

            # Step 3: account for mutual inhibition between MBONs
            MBON_updated = np.array(
                [
                    self.activation(MBON_activation[0] + self.w_nMBON_pMBON * MBON_activation[1]),
                    self.activation(MBON_activation[1] + self.w_pMBON_nMBON * MBON_activation[0]),
                ]
            )

            # Step 4: calculate the upwind drive
            upwind_drive = self.activation(np.dot(MBON_updated, np.array([self.w_pMBON_U, self.w_nMBON_U])))
            drives.append(upwind_drive)

        return drives

    def trial_plasticity(self, odor, reward):
        """
        A function to calculate the plasticity after a trial
        
        Parameters:
        -----------
        odor: int
            odor 1 or odor 2
        reward: int
            reward or punishment
        time_since_last_trial: float
            time since last trial in seconds (used for homeostatic plasticity, for now we set it to 1 arbitrary trial length)
        """

        # Step 0: calculate the KC activations
        if odor == 0:
            KC_activation = np.array([1, 0])
        elif odor == 1:
            KC_activation = np.array([0, 1])

        # Step 0.5: calculate the DAN activations
        if reward == 1:
            pDAN_activation = 1
            nDAN_activation = 0
        elif reward == -1:
            pDAN_activation = 0
            nDAN_activation = 1
        else:
            pDAN_activation = 0
            nDAN_activation = 0

        # Step 1: NO HOMEOSTATIC PLASTICITY
        self.w_KC_pMBON = self.w_KC_pMBON #+ (1 - self.w_KC_pMBON) * (1 - np.exp(-self.fr))
        self.w_KC_nMBON = self.w_KC_nMBON #+ (1 - self.w_KC_nMBON) * (1 - np.exp(-self.fr))

        # Step 2: calculate the MBON activations
        MBON_activation = np.array(
            [
                self.activation(np.dot(self.w_KC_pMBON, KC_activation)),
                self.activation(np.dot(self.w_KC_nMBON, KC_activation)),
            ]
        )

        # Step 3: account for mutual inhibition between MBONs
        MBON_updated = np.array(
            [
                self.activation(MBON_activation[0] + self.w_nMBON_pMBON * MBON_activation[1]),
                self.activation(MBON_activation[1] + self.w_pMBON_nMBON * MBON_activation[0]),
            ]
        )

        # Step 4: calculate the upwind drive
        upwind_drive = self.activation(np.dot(MBON_updated, np.array([self.w_pMBON_U, self.w_nMBON_U])))

        # Step 5: calculate the DAN activations
        pDAN_activation = self.activation(
            pDAN_activation
            + self.w_U_pDANs * upwind_drive
            + self.w_pMBON_pDANs * MBON_updated[0]
            - self.w_pMBON_pDANs  # to account for adaptation to typical DAN activation
            + self.w_nMBON_pDANs * MBON_updated[1]
            - self.w_nMBON_pDANs  # to account for adaptation to typical DAN activation
        )
        nDAN_activation = self.activation(
            nDAN_activation
            + self.w_U_nDANs * upwind_drive
            + self.w_pMBON_nDANs * MBON_updated[0]
            - self.w_pMBON_nDANs  # to account for adaptation to typical DAN activation
            + self.w_nMBON_nDANs * MBON_updated[1]
            - self.w_nMBON_nDANs  # to account for adaptation to typical DAN activation
        )

        # Step 6: calculate the plasticity and update the weights
        self.w_KC_pMBON = self.w_KC_pMBON \
                        - self.lr * nDAN_activation * KC_activation * self.w_KC_pMBON \
                        + self.fr * nDAN_activation * (1-KC_activation) * (1 - self.w_KC_pMBON)
        self.w_KC_nMBON = self.w_KC_nMBON \
                        - self.lr * pDAN_activation * KC_activation * self.w_KC_nMBON \
                        + self.fr * pDAN_activation * (1-KC_activation) * (1 - self.w_KC_nMBON)
        

        # Bound the weights
        self.w_KC_pMBON = np.clip(self.w_KC_pMBON, 0, 1)
        self.w_KC_nMBON = np.clip(self.w_KC_nMBON, 0, 1)

        # END of trial

    def get_weights(self):
        return self.w_KC_pMBON, self.w_KC_nMBON


# Helper Functions for Dataset


def get_valid_data(x):
    # check if x in a numpy array
    if not isinstance(x, np.ndarray):
        x_ = np.array(x)
    else:
        x_ = x.copy()
    x_ = x_[~np.isnan(x_)]
    x_ = x_[~np.isinf(x_)]
    x_ = x_[x_ >= 0]
    return x_


def get_split_data(choices, rewards):
    """
    Get the split choice and reward data for a given number of folds.
    
    Parameters
    ----------
    choices : numpy.ndarray
        The choice data.
    rewards : numpy.ndarray
        The reward data.
    """
    K = 2
    vals = []
    for i in range(K):
        cs_temp = choices[i::K]
        cs = []
        for c in cs_temp:
            cs += [get_valid_data(c)]
        rs_temp = rewards[i::K]
        rs = []
        for r in rs_temp:
            rs += [get_valid_data(r)]
        vals.append((cs, rs))
    return vals


def log_likelihood(predictions, observations):
    """
    Calculate the bernoulli log likelihood of the observation given the prediction.

    Parameters
    ----------
    predictions : numpy.ndarray
        The predictions.
    observations : numpy.ndarray
        The observations.
    """
    lls = []
    assert len(predictions) == len(observations), "The number of predictions and observations must be the same."
    for i in range(len(predictions)):
        lls += [np.sum(np.log(predictions[i]) * observations[i] + np.log(1 - predictions[i]) * (1 - observations[i]))]
    return lls


def normalized_log_likelihood(predictions, observations):
    """
    Calculate the normalized bernoulli log likelihood of the observation given the prediction.

    Parameters
    ----------
    predictions : numpy.ndarray
        The predictions.
    observations : numpy.ndarray
        The observations.
    """
    norm_lls = []
    assert len(predictions) == len(observations), "The number of predictions and observations must be the same."
    for i in range(len(predictions)):
        norm_lls += [
            np.exp(
                np.mean(np.log(predictions[i]) * observations[i] + np.log(1 - predictions[i]) * (1 - observations[i]))
            )
        ]
    return norm_lls


# create a parser object
parser = argparse.ArgumentParser(description="Fit a model to a dataset")
# add arguments with default values
parser.add_argument(
    "--algorithm", default="minimize", type=str, help="Algorithm to use (valid options: de, shgo, minimize)"
)
parser.add_argument(
    "--randomize",
    default=-1,
    type=int,
    help="Whether to randomize the data before fitting (valid options: -1 for no, positive integer for seed)",
)
parser.add_argument("--n_jobs", default=-1, type=int, help="Number of cores to use, -1 for all cores")
parser.add_argument(
    "--data",
    default="data/dmData_06-07-2023.ezip",
    type=str,
    help="Path to encrypted data file (if multifile, give the first file)",
)
parser.add_argument("--output", default="processed_data/dmData_14-09-2023/", type=str, help="Path to output directory")
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

Equivalence of Reinforcement Learning and Mushroom Body Models of Insect Learning
"""
print(start_string)


# check for valid algorithm
assert args.algorithm in ["de", "shgo", "minimize"], "Invalid algorithm"

# setup algorithm parameter defaults
algo_params = {}
if args.algorithm == "de":
    algo_params["popsize"] = 150
    algo_params["tol"] = 1e-3
    algo_params["disp"] = True
if args.algorithm == "shgo":
    algo_params["options"] = {"disp": True, "maxiter": 300}
    algo_params["iters"] = 1
if args.algorithm == "minimize":
    algo_params["tol"] = 1e-3

# setup the model fitting

# for now, we will ignore
# 1) cross modal plasticity so mu_inh = 0
# 2) cross modal excitation so fb_trans = 0
# ignored_params = {"mu_inh": 0.0, "fb_syn": 0.0}
ignored_params = {}


def loglik(params, choices, rewards):
    # returns the log likelihood of the data given the parameters
    fr, lr, up_dr, fb_trans, fb_up, mu_inh, fb_syn, pbn_asym = params
    # initialize the mushroom body
    sum_log_lik = 0

    def sub(i):
        MB = MushroomBody(
            fr=fr, lr=lr, up_dr=up_dr, fb_trans=fb_trans, fb_up=fb_up, mu_inh=mu_inh, fb_syn=fb_syn, pbn_asym=pbn_asym, **ignored_params
        )
        upwind_drives = []
        for j in range(len(choices[i])):
            upwind_drive = MB.upwind_drive()
            # apply softmax to upwind drives
            upwind_drive = np.exp(upwind_drive) / np.sum(np.exp(upwind_drive))
            # add to the list
            upwind_drives.append(upwind_drive)
            # randomly choose the odor
            MB.trial_plasticity(choices[i][j], rewards[i][j])
        upwind_drives = np.array(upwind_drives)
        # calculate the log likelihood (bernoulli)
        log_lik = np.sum(np.log(upwind_drives[:, 1]) * choices[i] + np.log(upwind_drives[:, 0]) * (1 - choices[i]))
        return log_lik

    # use joblib to parallelize the computation
    if args.n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    else:
        n_jobs = args.n_jobs
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(sub)(i) for i in range(len(choices)))
        sum_log_lik = np.sum(results)
    return -sum_log_lik


def fit_MB(choices, rewards):

    # set up the initial parameters and bounds
    eps = 1e-3
    params_init = np.array([5.0, 0.5, 5.0, 0.5, 0.5, 0.5, 0.5, 1.0])
    params_bounds = [
        (eps, 100),
        (eps, 1 - eps),
        (eps, 100),
        (eps, 1 - eps),
        (eps, 1 - eps),
        (eps, 1 - eps),
        (eps, 1 - eps),
        (eps, 1/eps),
    ]

    # run the optimization
    if args.algorithm == "de":
        from scipy.optimize import differential_evolution

        # callback function to print the current parameters
        def callback(xk, convergence):
            print(xk, loglik(xk, choices, rewards))

        res = differential_evolution(loglik, params_bounds, **algo_params, args=(choices, rewards), callback=callback)
    elif args.algorithm == "shgo":
        from scipy.optimize import shgo

        res = shgo(loglik, params_bounds, **algo_params, args=(choices, rewards))
    else:
        from scipy.optimize import minimize

        def callback(xk):
            print(xk, loglik(xk, choices, rewards))

        res = minimize(
            loglik, params_init, bounds=params_bounds, **algo_params, args=(choices, rewards), callback=callback
        )

    # return the results
    result_dict = {
        "fr": res.x[0],
        "lr": res.x[1],
        "up_dr": res.x[2],
        "fb_trans": res.x[3],
        "fb_up": res.x[4],
        "mu_inh": res.x[5],
        "fb_syn": res.x[6],
        "pbn_asym": res.x[7],
    }

    result_dict = {**result_dict, **ignored_params}

    return result_dict


def predict(choices, rewards, param):
    probs = []
    for i in range(len(choices)):
        MB = MushroomBody(**param)
        upwind_drives = []
        for j in range(len(choices[i])):
            upwind_drive = MB.upwind_drive()
            # apply softmax to upwind drives
            upwind_drive = np.exp(upwind_drive) / np.sum(np.exp(upwind_drive))
            # add to the list
            upwind_drives.append(upwind_drive)
            # randomly choose the odor
            MB.trial_plasticity(choices[i][j], rewards[i][j])
        upwind_drives = np.array(upwind_drives)
        probs.append(upwind_drives[:, 1])
    probs = np.array(probs, dtype=object)
    return probs


# check for valid number of cores
if args.n_jobs == -1:
    args.n_jobs = multiprocessing.cpu_count()
elif args.n_jobs > multiprocessing.cpu_count():
    print("Warning: More cores requested than available, using all cores")
    args.n_jobs = multiprocessing.cpu_count()
elif args.n_jobs < 1:
    print("Warning: Invalid number of cores requested, using 1 core")
    args.n_jobs = 1

# check for valid data path
if not (args.data.endswith(".ezip") or args.data.endswith(".ezip.000")) or not os.path.exists(args.data):
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
    os.makedirs(args.output + "/fit_results/")

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

algorithm = args.algorithm
n_jobs = args.n_jobs  # number of cores to use

# Print out the model and algorithm
start_str = "Initializing fit using {} algorithm with {} cores".format(algorithm, n_jobs)
print(start_str)
print("".join(["="] * len(start_str)))


# Set up the initial parameters and bounds
split_data = get_split_data(choices_full, rewards_full)

# Variables to store the results
params = []
train_probs = []
test_probs = []
train_log_liks = []
test_log_liks = []
train_norm_log_liks = []
test_norm_log_liks = []

# Fit the model to each split of the data
for choices, rewards in tqdm(split_data):
    # Fit the model
    param = fit_MB(choices, rewards)
    # Store the parameters
    params.append(param)
    # Compute the log likelihoods
    probs = predict(choices, rewards, param)
    train_probs.append(probs)
    train_log_liks.append(log_likelihood(probs, choices))
    train_norm_log_liks.append(normalized_log_likelihood(probs, choices))

# Test the model on the held-out data
for n, (choices, rewards) in tqdm(enumerate(split_data[::-1])):
    # Compute the log likelihoods
    probs = predict(choices, rewards, params[n])
    test_probs.append(probs)
    test_log_liks.append(log_likelihood(probs, choices))
    test_norm_log_liks.append(normalized_log_likelihood(probs, choices))

train_log_liks = np.concatenate(train_log_liks)
test_log_liks = np.concatenate(test_log_liks)
train_norm_log_liks = np.concatenate(train_norm_log_liks)
test_norm_log_liks = np.concatenate(test_norm_log_liks)

# Save to disk using compressed pickle format
if not os.path.isdir(args.output + "fit_results/"):
    os.mkdir(args.output + "fit_results/")

all_data = {
    "params": params,
    "train_log_liks": train_log_liks,
    "test_log_liks": test_log_liks,
    "train_norm_log_liks": train_norm_log_liks,
    "test_norm_log_liks": test_norm_log_liks,
    "train_probs": train_probs,
    "test_probs": test_probs,
}

# Dump to disk
with gzip.open(args.output + "fit_results/bivalent_depression_fit_results.pkl.gz", "wb") as f:
    pickle.dump(all_data, f)

