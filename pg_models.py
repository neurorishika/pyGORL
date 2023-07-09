## Setup a package for qlearning models

import numpy as np # for numerical operations
import scipy.optimize as opt # for numerical optimization

## STRUCTURE OF PARAMS
# params = [learning_param1, learning_param2, ..., learning_paramN, model_param1, model_param2, ..., model_paramM]
    
class VLPolicyGradient():
    """
    Single State Policy gradient model with logistic policy
    Serve as a base class for the different policy gradient models
    """
    def __init__(self,eps=1e-6,dx=0.01):
        """
        eps: a small number to prevent error for choice probabilities
        dx: the step size for numerical derivatives
        """
        self.eps = eps
        self.dx = dx

    def policy(self, params):
        """
        The policy function
        params: the parameters of the model
        """
        params = np.array(params)
        theta = params[0]
        logistic = lambda x: 1/(1+np.exp(-x))
        policy = np.array([logistic(theta), 1-logistic(theta)])
        policy = np.clip(policy, self.eps, 1-self.eps)
        return policy
    
    def log_policy(self, params):
        """
        The log policy function
        params: the parameters of the model
        """
        return np.log(self.policy(params))
    
    def del_log_policy(self, params):
        """
        The numerical derivative of the log policy function
        params: the parameters of the model
        """
        n_params = len(params)
        del_log_policy = np.zeros((n_params,2))
        for i in range(n_params):
            param_plus = params.copy()
            param_plus[i] = param_plus[i] + self.dx
            param_minus = params.copy()
            param_minus[i] = param_minus[i] - self.dx
            del_log_policy[i] = (self.log_policy(param_plus) - self.log_policy(param_minus))/(2*self.dx)
        return del_log_policy

    def policy_update(self, choice, reward, params):
        """
        The policy update function
        params: the parameters of the model
        choice: the choice made
        reward: the reward received
        """
        params = np.array(params)
        alpha = params[0]
        new_params = params.copy()
        try:
            new_params[self.param_props()['n_l']:] = params[self.param_props()['n_l']:] + alpha*reward*self.del_log_policy(params[self.param_props()['n_l']:])[:,int(choice)]
        except:
            new_params[self.param_props()['n_l']:] = params[self.param_props()['n_l']:]
        return new_params
    
    def policy_gradient_learning(self, choices, rewards, params):
        """
        The policy gradient learning function
        params: the parameters of the model
        choices: the choices made
        rewards: the rewards received
        """
        policy_params = np.zeros((len(choices)+1,len(params[self.param_props()['n_l']:])))
        policy_params[0] = params[self.param_props()['n_l']:]
        new_params = params.copy()
        for n, (choice, reward) in enumerate(zip(choices, rewards)):
            new_params = self.policy_update(choice, reward, new_params)
            policy_params[n+1] = new_params[self.param_props()['n_l']:]
        return policy_params
    
    def prob_choice(self, params, choices, rewards):
        """
        Return the probability of the choices given the parameters
        params: the parameters of the model
        choices: the choices made
        rewards: the rewards received
        """
        policy_params = self.policy_gradient_learning(choices, rewards, params)
        ps = np.zeros((len(choices)+1,2))
        for n, policy_param in enumerate(policy_params):
            ps[n] = self.policy(policy_param)
        return ps

    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['alpha', 'theta'],
            'suggested_bounds': [(0,1),(-10,10)],
            'suggested_init': [0.5,0.],
            'n_l': 1, # number of learning parameters
            'n_p': 1 # number of policy parameters
            }
        return param_props
    
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
    
    def fit_all(self, choices, rewards, params_init, lambda_reg=0, algo='de', **kwargs):
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
    
    def fit_subject(self, subject, choices, rewards, params_init, lambda_reg=0, algo='de', **kwargs):
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

    def fit_all_except(self, subject, choices, rewards, params_init, lambda_reg=0, algo='de', **kwargs):
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
    
    def fit_every_nth(self, start:int, K:int, choices, rewards, params_init, lambda_reg=0, algo='de', **kwargs):
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
    
# extend the policy gradient model to include a softmax function
class VSPolicyGradient(VLPolicyGradient):
    """
    Policy gradient model with softmax function
    """
    def __init__(self, eps=1e-6, dx=0.01):
        super().__init__(eps, dx)

    def policy(self, params):
        """
        Calculate the policy
        params: the parameters of the policy gradient model
        """
        # calculate the probability of choosing each action
        params = np.array(params)
        theta_0, theta_1 = params[0], params[1]
        softmax = lambda t0, t1: np.exp(t0)/(np.exp(t0)+np.exp(t1))
        policy = np.array([softmax(theta_0, theta_1), 1-softmax(theta_0, theta_1)])
        policy = np.clip(policy, self.eps, 1-self.eps)
        return policy
    
    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_l: the number of learning parameters
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['alpha', 'theta_0', 'theta_1'],
            'suggested_bounds': [(0,1),(-10,10),(-10,10)],
            'suggested_init': [0.5,0.,0.],
            'n_l': 1, # number of learning parameters
            'n_p': 2 # number of policy parameters
            }
        return param_props
    
class ACLPolicyGradient(VLPolicyGradient):
    """
    Actor-critic policy gradient model with logistic policy
    """
    def __init__(self, eps=1e-6, dx=0.01):
        super().__init__(eps, dx)
    
    def policy_update(self, choice, reward, params):
        """
        The policy update function
        params: the parameters of the model
        choice: the choice made
        reward: the reward received
        """
        params = np.array(params)
        alpha_p,alpha_q,qs = params[0], params[1], params[2:4]
        new_params = params.copy()
        try:
            new_params[self.param_props()['n_l']:] = params[self.param_props()['n_l']:] + alpha_p*qs[int(choice)]*self.del_log_policy(params[self.param_props()['n_l']:])[:,int(choice)]
            qs[int(choice)] = qs[int(choice)] + alpha_q*(reward-qs[int(choice)])
            new_params[2:4] = qs.copy()
        except:
            pass
        return new_params
    
    def policy_gradient_learning(self, choices, rewards, params):
        """
        The policy gradient learning function
        params: the parameters of the model
        choices: the choices made
        rewards: the rewards received
        """
        policy_params = np.zeros((len(choices)+1,len(params[self.param_props()['n_l']:])))
        policy_params[0] = params[self.param_props()['n_l']:]
        new_params = params.copy()
        for n, (choice, reward) in enumerate(zip(choices, rewards)):
            new_params = self.policy_update(choice, reward, new_params)
            policy_params[n+1] = new_params[self.param_props()['n_l']:]
        return policy_params
    
    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['alpha_policy', 'alpha_Q_value', 'Q_0', 'Q_1','theta'],
            'suggested_bounds': [(0,1),(0,1),(0,1),(0,1),(-10,10)],
            'suggested_init': [0.5,0.5,0.0,0.0,0.0],
            'n_l': 4, # number of learning parameters
            'n_p': 1 # number of policy parameters
            }
        return param_props

class AdvLPolicyGradient(VLPolicyGradient):
    """
    Advantage learning policy gradient model with logistic policy
    """
    def __init__(self, eps=1e-6, dx=0.01):
        super().__init__(eps, dx)
    
    def policy_update(self, choice, reward, params):
        """
        The policy update function
        params: the parameters of the model
        choice: the choice made
        reward: the reward received
        """
        params = np.array(params)
        alpha_p,alpha_v,v = params[0], params[1], params[2]
        new_params = params.copy()
        try:
            new_params[self.param_props()['n_l']:] = params[self.param_props()['n_l']:] + alpha_p*(reward-v)*self.del_log_policy(params[self.param_props()['n_l']:])[:,int(choice)]
            v = v + alpha_v*(reward-v)
            new_params[2] = v
        except:
            pass
        return new_params
    
    def policy_gradient_learning(self, choices, rewards, params):
        """
        The policy gradient learning function
        params: the parameters of the model
        choices: the choices made
        rewards: the rewards received
        """
        policy_params = np.zeros((len(choices)+1,len(params[self.param_props()['n_l']:])))
        policy_params[0] = params[self.param_props()['n_l']:]
        new_params = params.copy()
        for n, (choice, reward) in enumerate(zip(choices, rewards)):
            new_params = self.policy_update(choice, reward, new_params)
            policy_params[n+1] = new_params[self.param_props()['n_l']:]
        return policy_params
    
    def param_props(self):
        """
        Return the parameter properties
        names: the names of the parameters
        suggested_bounds: the suggested bounds for the parameters
        suggested_init: the suggested initial values for the parameters
        n_p: the number of policy parameters
        """
        param_props = {
            'names': ['alpha_policy', 'alpha_value', 'V','theta'],
            'suggested_bounds': [(0,1),(0,1),(0,1),(-10,10)],
            'suggested_init': [0.5,0.5,0.0,0.0],
            'n_l': 3, # number of learning parameters
            'n_p': 1 # number of policy parameters
            }
        return param_props