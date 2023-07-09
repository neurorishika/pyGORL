## Setup a package for qlearning models

import numpy as np # for numerical operations
import scipy.optimize as opt # for numerical optimization

## STRUCTURE OF PARAMS
# params = [q0_init, q1_init, qupdate_1, qupdate_2 ... qupdate_n, policy_1, policy_2 ... policy_n]

class QLearning():
    """
    Q learning model with softmax policy
    Serves as a base class for other Q learning models
    """

    def __init__(self, eps=1e-6):
        """
        Initialize the class
        eps: a small number to prevent errors for choice probabilities
        """
        self.eps = eps

    def q_update(self, q, choice, reward, params):
        """
        Update the q values
        q: the current q values
        choice: the choice made
        reward: the reward received
        params: the parameters of the q learning model
        """
        params = np.array(params)
        alpha_learn = params[0]
        q[choice] = q[choice] + alpha_learn*(reward - q[choice])
        return q

    def q_learning(self, choices, rewards, params):
        """
        Learn the q values
        choices: the choices made
        rewards: the rewards received
        params: the parameters of the q learning model
        """
        qs = np.zeros((len(choices)+1,2))
        qs[0,:] = params[:2]
        for i in range(len(choices)):
            qs[i+1,:] = self.q_update(qs[i,:].copy(),int(choices[i]), rewards[i], params[2:])
        return qs

    def policy(self,q,params):
        """
        Calculate the probability of choosing each action
        q: the q values
        params: the parameters of the q learning model
        """
        p = np.exp(np.clip(q/params[-1],-100,100))
        p = p / np.sum(p, axis=1)[:,None]
        return p

    def param_props(self):
        """
        Return the properties of the parameters
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_q: the number of q update parameters
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['q0_init', 'q1_init', 'alpha_learn', 'beta'],
            'suggested_bounds': [(0,1),(0,1),(0,1),(0.01,100)],
            'suggested_init': [0.5,0.5,0.5,1.],
            'n_q': 1,
            'n_p': 1
            }
        return param_props
    
    def prob_choice(self, choices, rewards, params):
        """
        Calculate the probability of choosing each action based on history
        choices: the choices made
        rewards: the rewards received
        params: the parameters of the q learning model
        """
        qs = self.q_learning(choices, rewards, params)
        ps = np.clip(self.policy(qs,params), self.eps, 1-self.eps) # clip to prevent errors
        return ps
    
    def regularizer(self, params):
        """
        Calculate the regularizer
        """
        return 0

    def nll(self, params, choices, rewards):
        """ 
        Calculate the negative log likelihood
        params: the parameters of the q learning model
        choices: the choices made
        rewards: the rewards received
        """
        lls = []
        for i in range(len(choices)):
            # remove after first nan
            if np.any(np.isnan(choices[i])):
                cs = choices[i][:np.argmax(np.isnan(choices[i]))]
            else:
                cs = choices[i]
            if np.any(np.isnan(rewards[i])):
                rs = rewards[i][:np.argmax(np.isnan(rewards[i]))]
            else:
                rs = rewards[i]
            assert len(cs) == len(rs), 'choices and rewards must be same length'
            # calculate the probability of each choice
            ps = self.prob_choice(cs, rs, params)[:-1,:]
            # calculate the log likelihood
            lls.append(np.sum(cs * np.log(ps[:,1]) + (1-cs) * np.log(ps[:,0])))
        # return the summation of negative log likelihood
        sum_lls = -np.sum(lls)
        return sum_lls
    
    def normll(self, params, choices, rewards):
        """
        Calculate the normalized log likelihood
        params: the parameters of the q learning model
        choices: the choices made
        rewards: the rewards received
        """
        normlls = []
        for i in range(len(choices)):
            # remove after first nan
            if np.any(np.isnan(choices[i])):
                cs = choices[i][:np.argmax(np.isnan(choices[i]))]
            else:
                cs = choices[i]
            if np.any(np.isnan(rewards[i])):
                rs = rewards[i][:np.argmax(np.isnan(rewards[i]))]
            else:
                rs = rewards[i]
            assert len(cs) == len(rs), 'choices and rewards must be same length'
            # calculate the probability of each choice
            ps = self.prob_choice(cs, rs, params)[:-1,:]
            # calculate the log likelihood
            normlls.append(np.exp(np.mean(cs * np.log(ps[:,1]) + (1-cs) * np.log(ps[:,0]))))
        # return the average of normalized log likelihood
        mean_normlls = np.mean(normlls)
        return mean_normlls
    
    def nll_reg(self, params, choices, rewards, lambda_reg):
        """
        Calculate the negative log likelihood with regularization
        params: the parameters of the q learning model
        choices: the choices made
        rewards: the rewards received
        lambda_reg: the regularization parameter
        """
        return self.nll(params, choices, rewards) + self.regularizer(params) * lambda_reg
    
    def fit_all(self, choices, rewards, params_init, lambda_reg=0, algo='shgo', **kwargs):
        """
        Fit the model to the data
        choices: the choices made
        rewards: the rewards received
        params_init: the initial parameters of the q learning model
        lambda_reg: the regularization parameter
        algo: the optimization algorithm to use
        kwargs: the keyword arguments for the optimization algorithm 
        """
        if algo == 'shgo':
            bounds = kwargs['bounds']
            # remove bounds from kwargs
            kwargs = {k: v for k, v in kwargs.items() if k != 'bounds'}
            print('Starting optimization with shgo algorithm')
            res = opt.shgo(
                self.nll_reg,
                args=(choices, rewards, lambda_reg),bounds=bounds,**kwargs)
        elif algo == 'de':
            bounds = kwargs['bounds']
            # remove bounds from kwargs
            kwargs = {k: v for k, v in kwargs.items() if k != 'bounds'}
            print('Starting optimization with differential evolution algorithm')
            res = opt.differential_evolution(
                self.nll_reg,
                args=(choices, rewards, lambda_reg),bounds=bounds,**kwargs)
        elif algo == 'basinhopping':
            bounds = kwargs['bounds']
            # remove bounds from kwargs
            kwargs = {k: v for k, v in kwargs.items() if k != 'bounds'}
            print('Starting optimization with basinhopping algorithm')
            res = opt.basinhopping(
                self.nll_reg,
                params_init,
                minimizer_kwargs={'args':(choices, rewards, lambda_reg)},**kwargs)
        elif algo == 'da':
            print('Starting optimization with dual annealing algorithm')
            res = opt.dual_annealing(
                self.nll_reg,
                args=(choices, rewards, lambda_reg),**kwargs)
        elif algo == 'minimize':
            assert 'randomize' in kwargs.keys(), 'Must specify whether to randomize initial parameters using "randomize" variable'
            assert 'n_restarts' in kwargs.keys(), 'Must specify number of restarts using "n_restarts" variable'
            res_list = []
            for X in range(kwargs['n_restarts']):
                print('Starting optimization with minimize. Iteration: {} of {}'.format(X+1,kwargs['n_restarts']))
                bounds = kwargs['bounds']
                if kwargs['randomize']:
                    # resample initial parameters
                    for i in range(len(params_init)):
                        params_init[i] = np.random.uniform(bounds[i][0],bounds[i][1])
                # remove randomize and bounds from kwargs
                kwargs_alt = {k: v for k, v in kwargs.items() if k not in ['randomize','bounds','n_restarts']}
                res = opt.minimize(
                    self.nll_reg,
                    params_init,
                    args=(choices, rewards, lambda_reg),**kwargs_alt)
                res_list.append(res)
            # find best result
            res = res_list[np.argmin([r.fun for r in res_list])]
        else:
            raise ValueError('Invalid algorithm')
        return res
    
    def fit_subject(self, subject, choices, rewards, params_init, lambda_reg=0, algo='shgo', **kwargs):
        """
        Fit the model to a single subject
        subject: the subject to fit the model to
        choices: the choices made
        rewards: the rewards received
        params_init: the initial parameters of the q learning model
        lambda_reg: the regularization parameter
        algo: the optimization algorithm to use
        kwargs: the keyword arguments for the optimization algorithm
        """
        res = self.fit_all(
            choices[subject:subject+1], rewards[subject:subject+1], params_init, lambda_reg, algo, **kwargs
            )
        return res

    def fit_all_except(self, subject, choices, rewards, params_init, lambda_reg=0, algo='shgo', **kwargs):
        """
        Fit the model to all subjects except one
        subject: the subject to exclude
        choices: the choices made
        rewards: the rewards received
        params_init: the initial parameters of the q learning model
        lambda_reg: the regularization parameter
        algo: the optimization algorithm to use
        kwargs: the keyword arguments for the optimization algorithm
        """
        assert subject < len(choices), 'subject must be less than number of subjects'
        res = self.fit_all(
            np.concatenate((choices[:subject],choices[subject+1:])),
            np.concatenate((rewards[:subject],rewards[subject+1:])),
            params_init, lambda_reg, algo, **kwargs
            )
        return res
    
    def fit_every_nth(self, start:int, K:int, choices, rewards, params_init, lambda_reg=0, algo='shgo', **kwargs):
        """
        Fit the model to every Kth subject
        start: the starting subject
        K: the number of subjects to skip
        choices: the choices made
        rewards: the rewards received
        params_init: the initial parameters of the q learning model
        lambda_reg: the regularization parameter
        algo: the optimization algorithm to use
        kwargs: the keyword arguments for the optimization algorithm
        """
        assert start < K, 'start must be less than K'
        choices = choices[start::K]
        rewards = rewards[start::K]
        res = self.fit_all(choices, rewards, params_init, lambda_reg, algo, **kwargs)
        return res

# extend the class to create new models

class HetQLearning(QLearning):
    """
    Heterogeneous q learning model with multiple time scales and softmax policy
    """
    def __init__(self, N_modules=2, mix_rule='weighted', eps=1e-6):
        """
        Initialize the model
        N_modules: the number of modules
        eps: a small number to prevent errors for choice probabilities
        """
        super().__init__(eps=eps)
        self.N_modules = N_modules
        assert mix_rule in ['weighted','max'], 'Invalid mixing rule'
        self.mix_rule = mix_rule

    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_q: the number of q update parameters per module
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['q0_init', 'q1_init'] + ['alpha_learn_%i' % i for i in range(self.N_modules)] + ['tau', 'beta'],
            'suggested_bounds': [(0,1),(0,1)] + [(0,1) for i in range(self.N_modules)] + [(1,30), (0.01,100)],
            'suggested_init': [0.5,0.5] + np.arange(1,self.N_modules+1)/(self.N_modules+1) + [7.,1.],
            'n_q': 1,
            'n_p': 2 
            }
        return param_props

    def q_learning(self, choices, rewards, params):
        """
        Perform q learning
        choices: the choices made
        rewards: the rewards received
        params: the parameters of the model
        """
        # define two sets of q values
        qs = np.zeros((len(choices)+1,2,self.N_modules))
        qs[:,0,:] = params[:2]

        n_q = self.param_props()['n_q']
        # update q values
        for i in range(len(choices)):
            for j in range(self.N_modules): # loop over modules
                qs[i+1,:,j] = self.q_update(qs[i,:,j].copy(),int(choices[i]), rewards[i], params[2+j*n_q:])

        # find the choice probabilities
        ps = np.clip(self.policy(qs,params), self.eps, 1-self.eps)[:-1,:,:]

        # find likelihood of each set of q values
        ll = choices[:,None] * np.log(ps[:,1,:]) + (1-choices[:,None]) * np.log(ps[:,0,:])
        
        # define the tau exponential kernel to apply to the likelihoods
        tau = params[-2]
        kernel = np.exp(-np.arange(len(choices))/ tau)
        cutoff_idx = np.where(kernel < 0.1)[0][0] if len(np.where(kernel < 0.1)[0]) > 0 else len(kernel)
        kernel = kernel[:cutoff_idx][::-1]
        kernel = kernel / np.sum(kernel)

        # apply the kernel convolution along the choice dimension
        ll = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='full'), 0, ll)[:len(choices),:]

        # apply the softmax
        w = np.exp(ll)
        w = w / np.sum(w, axis=1)[:,None]

        # if the mixing rule is max, make the weights binary
        if self.mix_rule == 'max':
            w = np.zeros_like(w)
            w[np.arange(len(w)), np.argmax(ll, axis=1)] = 1

        # calculate the weighted average of the q values
        out = np.zeros_like(qs[:-1,:,0])
        for i in range(self.N_modules):
            out += qs[:-1,:,i] * w[:,i][:,None]
        
        # append the first set of q values
        out = np.concatenate((qs.mean(axis=1)[0,:][None,:], out), axis=0)
        return out

class FQLearning(QLearning):
    """
    Q learning with forgetting and softmax policy
    """
    def q_update(self, q, choice, reward, params):
        """
        Redefine the q update function
        q: the q values
        choice: the choice made
        reward: the reward received
        params: the parameters of the model
        """
        params = np.array(params)
        alpha_learn, alpha_forget = params[0], params[1]
        q[choice] = q[choice] + alpha_learn*(reward - q[choice])
        q[1-choice] = (1-alpha_forget)*q[1-choice]
        return q

    # redefine the parameter properties
    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_q: the number of q update parameters
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['q0_init', 'q1_init', 'alpha_learn', 'alpha_forget', 'beta'],
            'suggested_bounds': [(0,1),(0,1),(0,1),(0,1),(0.01,100)],
            'suggested_init': [0.5,0.5,0.5,0.5,1.],
            'n_q': 2, # number of q update parameters
            'n_p': 1 # number of policy parameters
            }
        return param_props
    
class HetFQLearning(HetQLearning):
    """
    Heterogeneous q learning with forgetting and softmax policy
    """
    def q_update(self, q, choice, reward, params):
        """
        Redefine the q update function
        q: the q values
        choice: the choice made
        reward: the reward received
        params: the parameters of the model
        """
        params = np.array(params)
        alpha_learn, alpha_forget = params[0], params[1]
        q[choice] = q[choice] + alpha_learn*(reward - q[choice])
        q[1-choice] = (1-alpha_forget)*q[1-choice]
        return q

    # redefine the parameter properties
    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_q: the number of q update parameters per module
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['q0_init', 'q1_init'] + [val for pair in zip(['alpha_learn_%i' % i for i in range(self.N_modules)], 
                                                                   ['alpha_forget_%i' % i for i in range(self.N_modules)]) for val in pair] + ['tau', 'beta'],
            'suggested_bounds': [(0,1),(0,1)] + [val for pair in zip([(0,1) for i in range(self.N_modules)], 
                                                                     [(0,1) for i in range(self.N_modules)]) for val in pair] + [(1,30), (0.01,100)],
            'suggested_init': [0.5,0.5] + [val for pair in zip(np.arange(1,self.N_modules+1)/(self.N_modules+1),
                                                               np.arange(1,self.N_modules+1)[::-1]/(self.N_modules+1)) for val in pair] + [7.,1.],
            'n_q': 2,
            'n_p': 2
            }
        return param_props
    
class OSQLearning(QLearning):
    """
    Q learning with omission sensitivity and softmax policy
    """
    def q_update(self, q, choice, reward, params):
        """
        Redefine the q update function
        q: the q values
        choice: the choice made
        reward: the reward received
        params: the parameters of the model
        """
        params = np.array(params)
        alpha_learn, kappa = params[0], params[1]
        if reward != 0:
            q[choice] = q[choice] + alpha_learn*(reward - q[choice])
        else:
            q[choice] = q[choice] + alpha_learn*(kappa - q[choice])
        return q

    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_q: the number of q update parameters
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['q0_init', 'q1_init', 'alpha_learn', 'kappa', 'beta'],
            'suggested_bounds': [(0,1),(0,1),(0,1),(-1,1),(0.01,100)],
            'suggested_init': [0.5,0.5,0.5,0.0,1.],
            'n_q': 2,
            'n_p': 1
            }
        return param_props

class HetOSQLearning(HetQLearning):
    """
    Heterogeneous q learning with omission sensitivity and softmax policy
    """
    def q_update(self, q, choice, reward, params):
        """
        Redefine the q update function
        q: the q values
        choice: the choice made
        reward: the reward received
        params: the parameters of the model
        """
        params = np.array(params)
        alpha_learn, kappa = params[0], params[1]
        if reward != 0:
            q[choice] = q[choice] + alpha_learn*(reward - q[choice])
        else:
            q[choice] = q[choice] + alpha_learn*(kappa - q[choice])
        return q
    
    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_q: the number of q update parameters per module
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['q0_init', 'q1_init'] + [val for pair in zip(['alpha_learn_%i' % i for i in range(self.N_modules)], 
                                                                   ['kappa_%i' % i for i in range(self.N_modules)]) for val in pair] + ['tau', 'beta'],
            'suggested_bounds': [(0,1),(0,1)] + [val for pair in zip([(0,1) for i in range(self.N_modules)], 
                                                                     [(-1,1) for i in range(self.N_modules)]) for val in pair] + [(1,30), (0.01,100)],
            'suggested_init': [0.5,0.5] + [val for pair in zip(np.arange(1,self.N_modules+1)/(self.N_modules+1),
                                                               np.arange(1,self.N_modules+1)[::-1]/(self.N_modules+1)) for val in pair] + [7.,1.],
            'n_q': 2, # number of q update parameters per module
            'n_p': 2 # number of policy parameters
            }
        return param_props
    
class OSFQLearning(QLearning):
    """
    Q learning with omission sensitivity, forgetting and softmax policy
    """
    def q_update(self, q, choice, reward, params):
        """
        Redefine the q update function
        q: the q values
        choice: the choice made
        reward: the reward received
        params: the parameters of the model
        """
        params = np.array(params)
        alpha_learn, kappa, alpha_forget = params[0], params[1], params[2]
        if reward != 0:
            q[choice] = q[choice] + alpha_learn*(reward - q[choice])
        else:
            q[choice] = q[choice] + alpha_learn*(kappa - q[choice])
        q[1-choice] = (1-alpha_forget)*q[1-choice]
        return q

    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_q: the number of q update parameters
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['q0_init', 'q1_init', 'alpha_learn', 'kappa', 'alpha_forget', 'beta'],
            'suggested_bounds': [(0,1),(0,1),(0,1),(-1,1),(0,1),(0.01,100)],
            'suggested_init': [0.5,0.5,0.5,0.0,0.5,1.],
            'n_q': 3,
            'n_p': 1
            }
        return param_props
    
class HetOSFQLearning(HetQLearning):
    """
    Heterogeneous q learning with omission sensitivity, forgetting and softmax policy
    """
    def q_update(self, q, choice, reward, params):
        """
        Redefine the q update function
        q: the q values
        choice: the choice made
        reward: the reward received
        params: the parameters of the model
        """
        params = np.array(params)
        alpha_learn, kappa, alpha_forget = params[0], params[1], params[2]
        if reward != 0:
            q[choice] = q[choice] + alpha_learn*(reward - q[choice])
        else:
            q[choice] = q[choice] + alpha_learn*(kappa - q[choice])
        q[1-choice] = (1-alpha_forget)*q[1-choice]
        return q
    
    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_q: the number of q update parameters per module
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['q0_init', 'q1_init'] + [val for pair in zip(['alpha_learn_%i' % i for i in range(self.N_modules)],
                                                                   ['kappa_%i' % i for i in range(self.N_modules)],
                                                                   ['alpha_forget_%i' % i for i in range(self.N_modules)]) for val in pair] + ['tau', 'beta'],
            'suggested_bounds': [(0,1),(0,1)] + [val for pair in zip([(0,1) for i in range(self.N_modules)],
                                                                     [(-1,1) for i in range(self.N_modules)],
                                                                     [(0,1) for i in range(self.N_modules)]) for val in pair] + [(1,30), (0.01,100)],
            'suggested_init': [0.5,0.5] + [val for pair in zip(np.arange(1,self.N_modules+1)/(self.N_modules+1),
                                                                np.arange(1,self.N_modules+1)[::-1]/(self.N_modules+1),
                                                                np.arange(1,self.N_modules+1)/(self.N_modules+1)) for val in pair] + [7.,1.],
            'n_q': 3, # number of q update parameters per module
            'n_p': 2 # number of policy parameters
            }
        return param_props
    
class SOSFQLearning(QLearning):
    """
    Q learning with omission sensitivity, forgetting and softmax policy
    """
    def q_update(self, q, choice, reward, params):
        """
        Redefine the q update function
        q: the q values
        choice: the choice made
        reward: the reward received
        params: the parameters of the model
        """
        params = np.array(params)
        alpha_learn, kappa, alpha_forget = params[0], params[1], params[2]
        if reward != 0:
            q[choice] = q[choice] + alpha_learn*(reward - q[choice])
        else:
            q[choice] = q[choice] + alpha_learn*(kappa*q[choice] - q[choice])
        q[1-choice] = (1-alpha_forget)*q[1-choice]
        return q

    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_q: the number of q update parameters
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['q0_init', 'q1_init', 'alpha_learn', 'kappa', 'alpha_forget', 'beta'],
            'suggested_bounds': [(0,1),(0,1),(0,1),(-1,1),(0,1),(0.01,100)],
            'suggested_init': [0.5,0.5,0.5,0.0,0.5,1.],
            'n_q': 3, # number of q update parameters
            'n_p': 1 # number of policy parameters
            }
        return param_props

class HetSOSFQLearning(HetQLearning):
    """
    Heterogeneous q learning with omission sensitivity, forgetting and softmax policy
    """
    def q_update(self, q, choice, reward, params):
        """
        Redefine the q update function
        q: the q values
        choice: the choice made
        reward: the reward received
        params: the parameters of the model
        """
        params = np.array(params)
        alpha_learn, kappa, alpha_forget = params[0], params[1], params[2]
        if reward != 0:
            q[choice] = q[choice] + alpha_learn*(reward - q[choice])
        else:
            q[choice] = q[choice] + alpha_learn*(kappa*q[choice] - q[choice])
        q[1-choice] = (1-alpha_forget)*q[1-choice]
        return q
    
    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_q: the number of q update parameters per module
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['q0_init', 'q1_init'] + [val for pair in zip(['alpha_learn_%i' % i for i in range(self.N_modules)],
                                                                   ['kappa_%i' % i for i in range(self.N_modules)],
                                                                   ['alpha_forget_%i' % i for i in range(self.N_modules)]) for val in pair] + ['tau', 'beta'],
            'suggested_bounds': [(0,1),(0,1)] + [val for pair in zip([(0,1) for i in range(self.N_modules)],
                                                                     [(-1,1) for i in range(self.N_modules)],
                                                                     [(0,1) for i in range(self.N_modules)]) for val in pair] + [(1,30), (0.01,100)],
            'suggested_init': [0.5,0.5] + [val for pair in zip(np.arange(1,self.N_modules+1)/(self.N_modules+1),
                                                                np.arange(1,self.N_modules+1)[::-1]/(self.N_modules+1),
                                                                np.arange(1,self.N_modules+1)/(self.N_modules+1)) for val in pair] + [7.,1.],
            'n_q': 3, # number of q update parameters per module
            'n_p': 2 # number of policy parameters
            }
        return param_props