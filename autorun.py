# Automatically run variants of fit_kfold.py parallely while accounting for the CPU and memory usage

import psutil
import os
from subprocess import call

def get_cpu_usage():
    return psutil.cpu_percent()

def get_mem_usage():
    return psutil.virtual_memory().percent/100

print("Current System Status:")
print("CPU Usage: {}%, Memory Usage: {}%".format(get_cpu_usage(), get_mem_usage()))


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

while True:
    first_run = input("Run first experiment? (y/n): ") or 'n'
    if first_run in ['y','n','Y','N']:
        break
    else:
        print("Invalid input, please try again")

if first_run.lower() == 'y':
    with open('.tmp','w') as f:
        f.write('0')
else:
    # make sure the .tmp file exists
    if not os.path.exists('.tmp'):
        print("Error: .tmp file not found, starting from the first experiment")
        with open('.tmp','w') as f:
            f.write('0')

# read the .tmp file
with open('.tmp','r') as f:
    current_experiment = int(f.read())

# update the .tmp file
os.remove('.tmp')
with open('.tmp','w') as f:
    f.write(str(current_experiment+1))

# run the experiment in a new terminal
print("Running experiment {}...".format(current_experiment))
# check os type (windows or linux or mac)
if os.name == 'nt':
    # windows
    call(['start','cmd','/k'] + experiments_to_run[current_experiment])
elif os.name == 'posix':
    # linux or mac
    call(['gnome-terminal','--'] + experiments_to_run[current_experiment])
else:
    print("Error: OS not supported")
    exit(1)


