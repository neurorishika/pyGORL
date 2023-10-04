# Automatically run variants of scripts/fit_kfold.py parallely while accounting for the CPU and memory usage

import os
from subprocess import call

experiments_to_run = [
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "QL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--qc",
        "full",
    ],
    ["poetry", "run", "python", "scripts/fit_kfold.py", "--model", "FQL", "--data", "data/dmData_14-09-2023.ezip"],
    ["poetry", "run", "python", "scripts/fit_kfold.py", "--model", "OSQL", "--data", "data/dmData_14-09-2023.ezip"],
    ["poetry", "run", "python", "scripts/fit_kfold.py", "--model", "OSFQL", "--data", "data/dmData_14-09-2023.ezip"],
    ["poetry", "run", "python", "scripts/fit_kfold.py", "--model", "SOSFQL", "--data", "data/dmData_14-09-2023.ezip"],
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "HetQL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--n_modules",
        "2",
        "--mix_rule",
        "weighted",
        "--",
    ],
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "HetFQL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--n_modules",
        "2",
        "--mix_rule",
        "weighted",
    ],
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "HetOSQL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--n_modules",
        "2",
        "--mix_rule",
        "weighted",
    ],
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "HetOSFQL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--n_modules",
        "2",
        "--mix_rule",
        "weighted",
    ],
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "HetSOSFQL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--n_modules",
        "2",
        "--mix_rule",
        "weighted",
    ],
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "HetQL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--n_modules",
        "2",
        "--mix_rule",
        "max",
    ],
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "HetFQL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--n_modules",
        "2",
        "--mix_rule",
        "max",
    ],
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "HetOSQL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--n_modules",
        "2",
        "--mix_rule",
        "max",
    ],
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "HetOSFQL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--n_modules",
        "2",
        "--mix_rule",
        "max",
    ],
    [
        "poetry",
        "run",
        "python",
        "scripts/fit_kfold.py",
        "--model",
        "HetSOSFQL",
        "--data",
        "data/dmData_14-09-2023.ezip",
        "--n_modules",
        "2",
        "--mix_rule",
        "max",
    ],
    # ["poetry", "run", "python", "scripts/fit_kfold.py", "--model", "VLP", "--data", "data/dmData_14-09-2023.ezip"],
    # [
    #     "poetry",
    #     "run",
    #     "python",
    #     "scripts/fit_kfold.py",
    #     "--model",
    #     "ACLP",
    #     "--data",
    #     "data/dmData_14-09-2023.ezip",
    #     "--q_type",
    #     "q",
    # ],
    # [
    #     "poetry",
    #     "run",
    #     "python",
    #     "scripts/fit_kfold.py",
    #     "--model",
    #     "ACLP",
    #     "--data",
    #     "data/dmData_14-09-2023.ezip",
    #     "--q_type",
    #     "fq",
    # ],
    # [
    #     "poetry",
    #     "run",
    #     "python",
    #     "scripts/fit_kfold.py",
    #     "--model",
    #     "ACLP",
    #     "--data",
    #     "data/dmData_14-09-2023.ezip",
    #     "--q_type",
    #     "osfq",
    # ],
    # ["poetry", "run", "python", "scripts/fit_kfold.py", "--model", "AdvLP", "--data", "data/dmData_14-09-2023.ezip"],
]

while True:
    first_run = input("Run first experiment? (y/n): ") or "n"
    if first_run in ["y", "n", "Y", "N"]:
        break
    else:
        print("Invalid input, please try again")

if first_run.lower() == "y":
    with open(".tmp", "w") as f:
        f.write("0")
else:
    # make sure the .tmp file exists
    if not os.path.exists(".tmp"):
        print("Error: .tmp file not found, starting from the first experiment")
        with open(".tmp", "w") as f:
            f.write("0")

# read the .tmp file
with open(".tmp", "r") as f:
    current_experiment = int(f.read())

if current_experiment >= len(experiments_to_run):
    print("All experiments have been run, exiting...")
    os.remove(".tmp")
    exit()

# update the .tmp file
os.remove(".tmp")
with open(".tmp", "w") as f:
    f.write(str(current_experiment + 1))

# run the experiment in a new terminal
print("Running experiment {}...".format(current_experiment))
call(experiments_to_run[current_experiment])
