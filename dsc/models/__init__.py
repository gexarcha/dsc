#
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import numpy as np
from mpi4py import MPI

from itertools import combinations
from abc import ABCMeta, abstractmethod

import dsc.utils.parallel as parallel

from dsc.utils.datalog import dlog
# from dsc.em import Model
import itertools as itls


#=============================================================================#
# Abstract base class for component analysis models

class DModel():
    # __metaclass__ = ABCMeta
    """ Abstract base class for Sparse Coding models with binary latent variables
        and expectation tuncation (ET) based training scheme.

        This
    """

    def __init__(self, D, H, Hprime, gamma, states=np.array([-1., 0., 1.]), to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        """ Constructor for ET based Sparse Coding models with binary latent variables.

                        :param D: Dimension of observed data.
                        :type  D: int
                        :param H: Number of dictionary elements to learn.
                        :type  H: int
                        :param Hprime: ET approximation parameter: number of latent units to
                            choose during preselction
                        :type  Hprime: int
                        :param gamma: maximum number of active latent binary variables. This
                            parameter should be choosen to be larger than the expected sparseness
                            of the model to be learned.
                        :type  gamma: int

                        The set of model parameters of an CAModel derived model typically consist of::

                            model_param['W']:  dictionary elements (shape D times H)
                            model_param['pi']: prior activation probability for the observed variables.
                            model_param['sigma']: std-variance of observation noise

                        """
        self.comm = comm
        if not type(states) == np.ndarray:
            raise TypeError("DSC: states must be of type numpy.ndarray")
        if Hprime > H:
            raise Exception("Hprime must be less or equal to H")
        if gamma > Hprime:
            raise Exception("gamma must be less or equal to Hprime")
        self.to_learn = to_learn

        # Model meta-parameters
        self.D = D
        self.H = H
        self.Hprime = Hprime
        self.gamma = gamma
        self.states = states
        self.K = self.states.shape[0]
        self.K_0 = int(np.argwhere(states == 0.))

        # some sanity checks
        assert Hprime <= H
        assert gamma <= Hprime

        # Noise Policy
        tol = 1e-5
        self.noise_policy = {
            'W': (-np.inf, +np.inf, False),
            'pi': (tol,  1. - tol, False),
            'sigma': (0., +np.inf, False)
        }

        # Generate state-space list
        ss = np.empty((0, self.H), dtype=np.int8)
        for i in xrange(self.K):
            if (i == self.K_0):
                continue
            temp = np.eye(self.H, dtype=np.int8) * states[i]
            ss = np.concatenate((ss, temp))

        # all hidden vectors with a single active cause - for ternary 2*HxH
        self.SSM = ss[np.sum(np.abs(np.sign(ss)), 1) == 1]

        # all hidden vectors with more than one active cause
        # states_gen=self.get_states(self.states,self.Hprime,self.gamma)
        # self.SM=np.array([state for state in states_gen])
        self.SM = self.get_states(
            self.states, self.Hprime, self.gamma)

        # number of states with more than one active cause
        self.no_states = self.SM.shape[0]

        #
        self.state_abs = np.empty((self.K, self.no_states))
        for i in range(self.K):
            self.state_abs[i, :] = (
                self.SM == self.states[i]).sum(axis=1)
        self.state_abs[self.K_0, :] = self.H - \
            self.state_abs.sum(0) + self.state_abs[self.K_0, :]

    def generate_data(self, model_params, my_N, noise_on=True, gs=None, gp=None):
        D = self.D
        H = self.H
        states = self.states
        pi = model_params['pi']
        W = model_params['W'].T
        sigma = model_params['sigma']
        # Create output arrays, y is data, s is ground-truth
        y = np.zeros((my_N, D))
        s = np.zeros((my_N, H), dtype=np.int8)
        for n in xrange(my_N):
            if gs is None:
                s[n] = np.random.choice(states, size=H, replace=True, p=pi)
            else:
                assert gs.shape[0]==my_N
                if gp is None:
                    assert len(gs.shape)==2
                    s[n]=gs[n]
                else:
                    assert gp.shape[0]==my_N
                    assert gp.shape[1]==gs.shape[1]
                    s[n]=(gs[n]*gp[n]).sum(0)
                
            y[n] = np.dot(s[n], W)
        # Add noise according to the model parameters
        if noise_on:
            y += np.random.normal(scale=sigma, size=(my_N, D))
        # Build return structure
        return {'y': y, 's': s}

    def select_partial_data(self, anneal, my_data):
        """ Select a partial data-set from my_data and return it.

        The fraction of datapoints selected is determined by anneal['partial'].
        If anneal['partial'] is equal to either 1 or 0 the whole dataset will
        be returned.
        """
        partial = anneal['partial']

        if partial == 0 or partial == 1:              # partial == full data
            return my_data

        my_N, D = my_data['y'].shape
        my_pN = int(np.ceil(my_N * partial))

        if my_N == my_pN:                            # partial == full data
            return my_data

        # Choose subset...
        sel = np.random.permutation(my_N)[:my_pN]
        sel.sort()

        # Construct partial my_pdata...
        my_pdata = {}
        for key, val in my_data.items():
            my_pdata[key] = val[sel]

        return my_pdata

    def check_params(self, model_params):
        """ Perform a sanity check on the model parameters.
            Throw an exception if there are major violations;
            correct the parameter in case of minor violations
        """
        return model_params

    def step(self, anneal, model_params, my_data):
        """ Perform an EM-step """

        # Noisify model parameters
        model_params = self.noisify_params(model_params, anneal)

        # Sanity check model parameters
        model_params = self.check_params(model_params)

        # For partial EM-step: select batch
        my_pdata = self.select_partial_data(anneal, my_data)

        # Annotate partial dataset with hidden-state candidates
        my_pdata = self.select_Hprimes(model_params, my_pdata)

        # Do E-step and calculate joint-probabilities
        my_joint_prob = self.E_step(anneal, model_params, my_pdata)

        # Use joint-probabilities to derive new parameter set
        new_model_params = self.M_step(
            anneal, model_params, my_joint_prob, my_pdata)

        # Calculate Objecive Function
        #[Q, theo_Q] = self.objective(my_joint_prob, model_params, my_pdata)
        #dlog.append('Q', Q)
        #dlog.append('theo_Q', theo_Q)

        # Log iboth model parameters and annealing parameters
        dlog.append_all(new_model_params)
        dlog.append_all(anneal.as_dict())

        return new_model_params

    def standard_init(self, data):
        """ Standard onitial estimation for model parameters.

        This implementation

        *W* and *sigma*.



        each *W* raw is set to the average over the data plus WGN of mean zero
        and var *sigma*/4. *sigma* is set to the variance of the data around
        the computed mean. *pi* is set to 1./H . Returns a dict with the
        estimated parameter set with entries "W", "pi" and "sigma".
        """
        comm = self.comm
        H = self.H
        my_y = data['y']
        my_N, D = my_y.shape

        assert D == self.D

        # Calculate averarge W
        W_mean = parallel.allmean(my_y, axis=0, comm=comm)     # shape: (D, )

        # Calculate data variance
        # import ipdb;ipdb.set_trace()  ######### Break Point ###########

        sigma_sq = parallel.allmean((my_y - W_mean)**2, axis=0, comm=comm)  # shape: (D, )
        sigma_init = np.sqrt(sigma_sq).sum()/D                         # scalar

        # Initial W
        noise = sigma_init / 4.
        # noise = sigma_init * 4.
        # shape: (H, D)`
        # W_init = W_mean[:, None] + np.random.laplace(scale=noise, size=[D, H])
        W_init = W_mean[:, None] + np.random.normal(scale=noise, size=[D, H])

        sparsity = 1. - (1./H) 
        pi_init = np.random.rand(self.K - 1)
        pi_init = (1 - sparsity) * pi_init / pi_init.sum()
        pi_init = np.insert(pi_init, self.K_0, sparsity)

        # Create and set Model Parameters, W columns have the same average!
        model_params = {
            'W': W_init,
            'pi': pi_init,
            'sigma': sigma_init
        }

        return model_params

    def inference(self, anneal, model_params, my_data, no_maps=10):
        W = model_params['W']
        my_y = my_data['y']
        D, H = W.shape
        my_N, D = my_y.shape

        # Prepare return structure
        res = {
            's': np.zeros((my_N, no_maps, H), dtype=np.int),
            'p': np.zeros((my_N, no_maps))
        }

        if 'candidates' not in my_data:
            my_data = self.select_Hprimes(model_params, my_data)
            my_cand = my_data['candidates']

        my_suff_stat = self.E_step(anneal, model_params, my_data)
        my_logpj = my_suff_stat['logpj']
        my_corr = my_logpj.max(axis=1)           # shape: (my_N,)
        my_logpjc = my_logpj - my_corr[:, None]    # shape: (my_N, no_states)
        my_pjc = np.exp(my_logpjc)              # shape: (my_N, no_states)
        my_denomc = my_pjc.sum(axis=1)             # shape: (my_N)

        idx = np.argsort(my_logpjc, axis=-1)[:, ::-1]
        for n in xrange(my_N):                                   # XXX Vectorize XXX
            for m in xrange(no_maps):
                this_idx = idx[n, m]
                res['p'][n, m] = my_pjc[n, this_idx] / my_denomc[n]
                if this_idx == 0:
                    pass
                elif this_idx < ((self.K-1)*H + 1):
                    s_prime = self.SSM[this_idx-1,:]
                    res['s'][n, m, :] = s_prime
                else:
                    s_prime = self.SM[this_idx - (self.K-1)*H - 1,:]
                    res['s'][n, m, my_cand[n, :]] = s_prime

        return res


    def get_states(self, states, Hprime, gamma):
        tmp = np.array(list(itls.product(states, repeat=Hprime)))
        c1 = (np.sum(tmp != 0, 1) <= gamma) * (np.sum(tmp != 0, 1) > 1)
        return tmp[c1]
