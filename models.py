## Setup a package for qlearning models

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

## STRUCTURE OF PARAMS
# params = [q0_init, q1_init, qupdate_1, qupdate_2 ... qupdate_n, policy_1, policy_2 ... policy_n]

class QLearning():

    # initialize the class
    def __init__(self, eps=1e-6):
        self.eps = eps

    # define the q update function
    def q_update(self, q, choice, reward, params):
        params = np.array(params)
        alpha = params[0]
        q[choice] = q[choice] + alpha*(reward - q[choice])
        return q

    # define the q learning function
    def q_learning(self, choices, rewards, params):
        qs = np.zeros((len(choices)+1,2))
        qs[0,:] = params[:2]
        for i in range(len(choices)):
            qs[i+1,:] = self.q_update(qs[i,:],int(choices[i]), rewards[i], params[2:])
        return qs

    # define the policy function
    def policy(self,q,params):
        p = np.exp(np.clip(q/params[-1],-100,100))
        p = p / np.sum(p, axis=1)[:,None]
        return p

    # define parameter names
    def param_props(self):
        param_props = {
            'names': ['q0_init', 'q1_init', 'alpha', 'beta'],
            'suggested_bounds': [(0,1),(0,1),(0,1),(0.01,100)],
            'suggested_init': [0.5,0.5,0.5,1.],
            'n_q': 1, # number of q update parameters
            'n_p': 1 # number of policy parameters
            }
        return param_props
    
    # calculate the probability of choosing each action based on history
    def prob_choice(self, choices, rewards, params):
        qs = self.q_learning(choices, rewards, params)
        ps = np.clip(self.policy(qs,params), self.eps, 1-self.eps)
        return ps
    
    # define the regularizer
    def regularizer(self, params):
        return 0

    # calculate the negative log likelihood
    def nll(self, params, choices, rewards):
        lls = []
        for i in range(len(choices)):
            ps = self.prob_choice(choices[i], rewards[i], params)[:-1,:]
            lls.append(np.sum(choices[i] * np.log(ps[:,1]) + (1-choices[i]) * np.log(ps[:,0])))
        return -np.sum(lls)
    
    # calculate the regularized negative log likelihood
    def nll_reg(self, params, choices, rewards, lambda_reg):
        return self.nll(params, choices, rewards) + self.regularizer(params) * lambda_reg
    
    # fit the model to all subjects
    def fit_all(self, choices, rewards, params_init, lambda_reg=0, algo='shgo', **kwargs):
        if algo == 'shgo':
            res = opt.shgo(
                self.nll_reg,
                args=(choices, rewards, lambda_reg),**kwargs)
        elif algo == 'de':
            res = opt.differential_evolution(
                self.nll_reg,
                args=(choices, rewards, lambda_reg),**kwargs)
        elif algo == 'basinhopping':
            res = opt.basinhopping(
                self.nll_reg,
                params_init,
                minimizer_kwargs={'args':(choices, rewards, lambda_reg)},**kwargs)
        elif algo == 'minimize':
            res = opt.minimize(
                self.nll_reg,
                params_init,
                args=(choices, rewards, lambda_reg),**kwargs)
        else:
            raise ValueError('Invalid algorithm')
        return res
    
    # define a function to fit the model to a single subject
    def fit_subject(self, subject, choices, rewards, params_init, lambda_reg=0, algo='shgo', **kwargs):
        res = self.fit_all(
            choices[subject:subject+1], rewards[subject:subject+1], params_init, lambda_reg, algo, **kwargs
            )
        return res

    # define a function to fit the model to all subjects except one
    def fit_all_except(self, subject, choices, rewards, params_init, lambda_reg=0, algo='shgo', **kwargs):
        res = self.fit_all(
            np.concatenate((choices[:subject],choices[subject+1:])),
            np.concatenate((rewards[:subject],rewards[subject+1:])),
            params_init, lambda_reg, algo, **kwargs
            )
        return res

# extend the class to create new models

class HetQLearning(QLearning):

    # redefine the init function
    def __init__(self, N_modules=2, eps=1e-6):
        self.N_modules = N_modules
        self.eps = eps

    # redefine the parameter properties
    def param_props(self):
        param_props = {
            'names': ['q0_init', 'q1_init'] + ['alpha_%i' % i for i in range(self.N_modules)] + ['tau', 'beta'],
            'suggested_bounds': [(0,1),(0,1)] + [(0,1) for i in range(self.N_modules)] + [(1,30), (0.01,100)],
            'suggested_init': [0.5,0.5] + np.arange(1,self.N_modules+1)/(self.N_modules+1) + [7.,1.],
            'n_q': 1, # number of q update parameters per module
            'n_p': 2 # number of policy parameters
            }
        return param_props

    # redefine the q learning function
    def q_learning(self, choices, rewards, params):
        # define two sets of q values
        qs = np.zeros((len(choices)+1,2,self.N_modules))
        qs[:,0,:] = params[:2]
        n_q = self.param_props()['n_q']
        for i in range(len(choices)):
            for j in range(self.N_modules):
                qs[i+1,:,j] = self.q_update(qs[i,:,j],int(choices[i]), rewards[i], params[2+j*n_q:])
        # find likelihood of each set of q values
        ps = np.clip(self.policy(qs,params), self.eps, 1-self.eps)[:-1,:,:]
        ll = choices[:,None] * np.log(ps[:,1,:]) + (1-choices[:,None]) * np.log(ps[:,0,:])
        # define the tau exponential kernel
        tau = params[-2]
        kernel = np.exp(np.arange(len(choices))/ tau)
        cutoff_idx = np.where(kernel < 0.1)[0][0] if len(np.where(kernel < 0.1)[0]) > 0 else len(kernel)
        kernel = kernel[:cutoff_idx][::-1]
        kernel = kernel / np.sum(kernel)
        # apply the kernel convolution along the choice dimension
        ll = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='full'), 0, ll)[:len(choices),:]
        # apply the softmax
        w = np.exp(ll)
        w = w / np.sum(w, axis=1)[:,None]
        # calculate the weighted average of the q values
        out = np.zeros_like(qs[:-1,:,0])
        for i in range(self.N_modules):
            out += qs[:-1,:,i] * w[:,i][:,None]
        # append the first set of q values
        out = np.concatenate((qs.mean(axis=1)[0,:][None,:], out), axis=0)
        return out

class FQLearning(QLearning):
    # redefine the q update function
    def q_update(self, q, choice, reward, params):
        params = np.array(params)
        alpha, kappa = params[0], params[1]
        q[choice] = q[choice] + alpha*(reward - q[choice])
        q[1-choice] = (1-kappa)*q[1-choice]
        return q

    # redefine the parameter properties
    def param_props(self):
        param_props = {
            'names': ['q0_init', 'q1_init', 'alpha', 'kappa', 'beta'],
            'suggested_bounds': [(0,1),(0,1),(0,1),(0,1),(0.01,100)],
            'suggested_init': [0.5,0.5,0.5,0.5,1.],
            'n_q': 2, # number of q update parameters
            'n_p': 1 # number of policy parameters
            }
        return param_props
    
class HetFQLearning(HetQLearning):
    # redefine the q update function
    def q_update(self, q, choice, reward, params):
        params = np.array(params)
        alpha, kappa = params[0], params[1]
        q[choice] = q[choice] + alpha*(reward - q[choice])
        q[1-choice] = (1-kappa)*q[1-choice]
        return q

    # redefine the parameter properties
    def param_props(self):
        param_props = {
            'names': ['q0_init', 'q1_init'] + [val for pair in zip(['alpha_%i' % i for i in range(self.N_modules)], 
                                                                   ['kappa_%i' % i for i in range(self.N_modules)]) for val in pair] + ['tau', 'beta'],
            'suggested_bounds': [(0,1),(0,1)] + [val for pair in zip([(0,1) for i in range(self.N_modules)], 
                                                                     [(0,1) for i in range(self.N_modules)]) for val in pair] + [(1,30), (0.01,100)],
            'suggested_init': [0.5,0.5] + [val for pair in zip(np.arange(1,self.N_modules+1)/(self.N_modules+1),
                                                               np.arange(1,self.N_modules+1)[::-1]/(self.N_modules+1)) for val in pair] + [7.,1.],
            'n_q': 2, # number of q update parameters per module
            'n_p': 2 # number of policy parameters
            }
        return param_props