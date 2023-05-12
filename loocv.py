import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from joblib import Parallel, delayed
from tqdm import tqdm
from models import FQLearning
import pickle

# Importing the dataset
choices_full = np.loadtxt('full_action_set.csv', delimiter=',')
rewards_full = np.loadtxt('full_reward_set.csv', delimiter=',')

assert len(choices_full) == len(rewards_full), "Choices and rewards are not the same length"
N = len(choices_full)

n_jobs = 2 # number of cores to use (free to change this)

# Set up the model
model = FQLearning()
print("Parameters to be estimated: ", ", ".join(model.param_props()['names']))
params_init = model.param_props()['suggested_init']
params_bounds = model.param_props()['suggested_bounds']

for i in range(N//n_jobs):
    results = Parallel(n_jobs=n_jobs)(
        delayed(model.fit_all_except)(
            subject, choices_full, rewards_full, params_init, lambda_reg=0,
            bounds=params_bounds,
            algo='shgo',
            options={'disp':True}
            # maxiter=1000, popsize=100, tol=1e-3, disp=True
            ) for subject in tqdm(range(n_jobs*i, n_jobs*(i+1)))
            )

    # Save the results as a pickle file
    with open(f'loocv_results_{i}.pkl', 'wb') as f:
        pickle.dump(results, f)

results = []
# load and collate all results
for i in range(12):
    with open(f'loocv_results_{i}.pkl','rb') as f:
        results = results + pickle.load(f)

# save a single file
with open('FQL_loocv_shgo.pkl','wb') as f:
    pickle.dump(results,f)






