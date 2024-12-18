# PyGORL: Python based fitting of Globally Optimized Reinforcement Learning algorithms

Author: [Rishika Mohanta](https://neurorishika.github.io)

This repository contains the code fitting multiple variations of Reinforcement Learning for a 1-state Markov Decision Process (MDP) based learning problem such as a 2-armed Bandit Task. Since it is a 1-state task, we decided to make value decisions on the basis of R_t rather than G_t and as a result there is no discounting factor. The policy function is assumed to be softmax for all value based algorithms and logistic for all policy gradient based algorithms.

As of right now the following algorithms are implemented:
- [x] Naive Q-learning (QL)
- [x] Q-learning with forgetting (FQL)
- [x] Q-learning with omission-sensitivity (OSQL)
- [x] Q-learning with omission-sensitivity and forgetting (OSFQL)
- [x] Q-learning with scaled-omission-sensitivity and forgetting (SOSFQL)
- [x] Variants of above algorithms with heterogenous rates with X independent modules mixed with weights given by softmax of likelihood over past observations (HetXQL, HetXFQL, HetXOSQL, HetXOSFQL, HetXSOSFQL)
- [x] Vanilla Policy Gradient (VPL)
- [x] Actor-Critic Policy Gradient with omission-sensitivity and forgetting variants (ACPL, ACPL-F, ACPL-OSF)
- [x] Advantage Actor-Critic Policy Gradient (AdvPL)

Data is generated from a 2-armed bandit task performed by GR64f-UAS/CsChrimson-Gal4 flies with optogenetic rewards. Data is protected using encryption and can be decrypted using the key provided by the authors. 

The data is stored in the `data` folder.

USE [Poetry](https://python-poetry.org/) to install dependencies.

For fitting the models, call: `poetry run python fit_kfold.py` and use `--help` for more information on the arguments. To fit all models on a Windows machine, use `autorun.bat' to start a new shell process for each model. Every time you run the batch file, it spawns the next job.