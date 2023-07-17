import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from joblib import Parallel, delayed
from tqdm import tqdm
from pygorl.cogq import *
from pygorl.cogpolicy import *
import shutil
import pickle
import os

# Importing the dataset
choices_full = np.loadtxt('data/kaitlyn2023/full_action_set.csv', delimiter=',')
rewards_full = np.loadtxt('data/kaitlyn2023/full_reward_set.csv', delimiter=',')

assert len(choices_full) == len(rewards_full), "Choices and rewards are not the same length"
N = len(choices_full)

n_jobs = 1 # number of cores to use (free to change this)

# Set up the model
model = FQLearning()
model_name = 'FQL'
algorithm = 'minimize'
print("Parameters to be estimated: ", ", ".join(model.param_props()['names']))
params_init = model.param_props()['suggested_init']
params_bounds = model.param_props()['suggested_bounds']

if not os.path.exists('tmp'):
    os.mkdir('tmp')
if not os.path.exists(f'tmp/{model_name}'):
    os.mkdir(f'tmp/{model_name}')

for i in range(N//n_jobs):
    print(f"Running LOOCV iteration {i+1} of {N//n_jobs}")
    results = Parallel(n_jobs=n_jobs)(
        delayed(model.fit_all_except)(
            subject, choices_full, rewards_full, params_init, lambda_reg=0,
            bounds=params_bounds,
            algo=algorithm,
            randomize=True
            # options={'disp':True,'maxiter':300}, iters=1
            # maxiter=1000, popsize=100, tol=1e-3, disp=True
            ) for subject in tqdm(range(n_jobs*i, n_jobs*(i+1)))
            )

    # Save the results as a pickle file
    with open(f'tmp/{model_name}/loocv_results_{i}.pkl', 'wb') as f:
        pickle.dump(results, f)

results = []
# load and collate all results
for i in range(N//n_jobs):
    with open(f'tmp/{model_name}/loocv_results_{i}.pkl','rb') as f:
        results = results + pickle.load(f)

from datetime import datetime
dt = datetime.now().strftime('%Y%m%d')

# save a single file
with open(f'fitted_models/kaitlyn2023/{model_name}_loocv_{algorithm}_{dt}.pkl','wb') as f:
    pickle.dump(results,f)

# delete the individual files
for i in range(N//n_jobs):
    os.remove(f'tmp/{model_name}/loocv_results_{i}.pkl')

# delete the tmp directory
shutil.rmtree('tmp')






