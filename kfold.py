import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from joblib import Parallel, delayed
from tqdm import tqdm
from models import *
import pickle
import os

# Importing the dataset
choices_full = np.loadtxt('data/kaitlyn2023/full_action_set.csv', delimiter=',')
rewards_full = np.loadtxt('data/kaitlyn2023/full_reward_set.csv', delimiter=',')

assert len(choices_full) == len(rewards_full), "Choices and rewards are not the same length"
N = len(choices_full)

K = 2 # number of folds
n_jobs = 2 # number of cores to use (free to change this)

# Set up the model
model = OSQLearning()
model_name = 'OSQL'
algorithm = 'de'
print("Parameters to be estimated: ", ", ".join(model.param_props()['names']))
params_init = model.param_props()['suggested_init']
params_bounds = model.param_props()['suggested_bounds']

if not os.path.exists('temp'):
    os.mkdir('temp')
if not os.path.exists(f'temp/{model_name}'):
    os.mkdir(f'temp/{model_name}')

results = Parallel(n_jobs=n_jobs)(
    delayed(model.fit_every_nth)(
        start, K,
        choices_full, rewards_full, params_init, lambda_reg=0,
        bounds=params_bounds,
        algo=algorithm,   
        # randomize=True, n_restarts=20 # for minimize
        # options={'disp':True,'maxiter':300}, iters=1 # for shgo
        maxiter=500, popsize=100, tol=1e-3, disp=True # for differential_evolution
        ) for start in tqdm(range(n_jobs))
        )

from datetime import datetime
dt = datetime.now().strftime('%Y%m%d')

# save a single file
with open(f'fitted_models/kaitlyn2023/{model_name}_{K}cv_{algorithm}_{dt}.pkl','wb') as f:
    pickle.dump(results,f)







