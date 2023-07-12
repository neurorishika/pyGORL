# Automatically run variants of fit_kfold.py parallely while accounting for the CPU and memory usage

import psutil
import os
from subprocess import Popen
import time

# Set the maximum CPU and memory usage
max_cpu = 0.8
max_mem = 0.8

def get_cpu_usage():
    return psutil.cpu_percent()

def get_mem_usage():
    return psutil.virtual_memory().percent/100

experiments_to_run = [
    ['python','fit_kfold.py','--model', 'QL','--filterdate','2023-06-10'],
    ['python','fit_kfold.py','--model', 'FQL','--filterdate','2023-06-10'],
    ['python','fit_kfold.py','--model', 'OSQL','--filterdate','2023-06-10'],
    ['python','fit_kfold.py','--model', 'OSFQL','--filterdate','2023-06-10'],
    ['python','fit_kfold.py','--model', 'SOSFQL','--filterdate','2023-06-10'],
    ['python','fit_kfold.py','--model', 'HetQL','--filterdate','2023-06-10','--n_modules','2'],
    ['python','fit_kfold.py','--model', 'HetFQL','--filterdate','2023-06-10','--n_modules','2'],
    ['python','fit_kfold.py','--model', 'HetOSQL','--filterdate','2023-06-10','--n_modules','2'],
    ['python','fit_kfold.py','--model', 'HetOSFQL','--filterdate','2023-06-10','--n_modules','2'],
    ['python','fit_kfold.py','--model', 'HetSOSFQL','--filterdate','2023-06-10','--n_modules','2'],
    ['python','fit_kfold.py','--model', 'VLP','--filterdate','2023-06-10'],
    ['python','fit_kfold.py','--model', 'ACLP','--filterdate','2023-06-10','--q_type','q'],
    ['python','fit_kfold.py','--model', 'ACLP','--filterdate','2023-06-10','--q_type','fq'],
    ['python','fit_kfold.py','--model', 'ACLP','--filterdate','2023-06-10','--q_type','osfq'],
    ['python','fit_kfold.py','--model', 'AdvLP','--filterdate','2023-06-10'],
]

run_till = -1
processes = []
while True:
    # check if cpu and memory usage is below the threshold
    if get_cpu_usage() < max_cpu and get_mem_usage() < max_mem:
        # check if all experiments have been run
        if run_till == len(experiments_to_run)-1:
            break
        # run the next experiment
        run_till += 1
        print("Running experiment ",run_till)
        # run the experiment by passing the arguments to Popen and fetch the process id
        processes.append(Popen(experiments_to_run[run_till]))
    else:
        # wait for 10 seconds
        time.sleep(10)

# wait for all processes to finish
for p in processes:
    p.wait()