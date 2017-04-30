#
#  Lincense: Academic Free License (AFL) v3.0
#



import numpy as np
from mpi4py import MPI
import itertools as itls
from scipy.misc import logsumexp


import dsc.utils.parallel as parallel
from dsc.utils.parallel import pprint as pp

from dsc.utils.datalog import dlog
from dsc.models import DModel

from scipy.special import gammaln


def multinom2(a,b):
    """
    :param a: array
    :param b: array
    :return:np.exp(gammaln(a)-gammaln(b).sum())
    """
    return np.exp(gammaln(a+1)-gammaln(b+1).sum())


class DSC_ET(DModel):
    def __init__(self, D, H, Hprime, gamma,states=np.array([-1.,0.,1.]), to_learn=['W', 'pi', 'sigma'], comm=MPI.COMM_WORLD):
        DModel.__init__(self, D, H, Hprime, gamma,states, to_learn, comm)

    def check_params(self, model_params):
        """ Sanity check.

        Sanity-check the given model parameters. Raises an exception if
        something is severely wrong.
        """
        W     = model_params['W']
        pies  = model_params['pi']
        sigma = model_params['sigma']

        assert np.isfinite(W).all()      # check W

        assert np.isfinite(pies).all()   # check pies
        # assert pies.sum()<=1.
        # assert (pies>=0).all()
        #assert pies <= 1.

        assert np.isfinite(sigma).all()  # check sigma
        assert sigma >= 0.

        return model_params


    def select_Hprimes(self, model_params, data,):
        """
        Return a new data-dictionary which has been annotated with
        a data['candidates'] dataset. A set of self.Hprime candidates
        will be selected.
        """
        my_N, D   = data['y'].shape
        H         = self.H
        SSM        = self.SSM
        nss = SSM.shape[0]
        candidates= np.zeros((my_N, self.Hprime), dtype=np.int)
        W         = model_params['W'].T
        pi        = model_params['pi']
        sigma     = model_params['sigma']

        # Precompute
        pre1     = -1./2./sigma/sigma
        l_pis=np.zeros((self.H*(self.K-1)))
        c=0
        for i in range(self.K):
            if i ==self.K_0:
                continue
            l_pis[c*H:(c+1)*H]+=np.log(pi[i]) + (self.H-1)*np.log(pi[self.K_0])
            c+=1
            # l_pis+=self.state_abs[i]*pi[i]
        # Allocate return structures
        F = np.empty( [my_N, nss] )
        for n in range(my_N):
            y    = data['y'][n,:]

            Wbar = np.dot(SSM,W)
            log_prod_joint = pre1 * (((Wbar-y)**2).sum(axis=1))
            F[n] = log_prod_joint
            F__= F[n]+l_pis
            sort_prob_ind=np.mod(np.argsort(F__),H)[::-1]
            Fu,Si = np.unique(sort_prob_ind,return_index=True)
            cand = Fu[np.argsort(Si)][:self.Hprime]
            candidates[n]=cand
        data['candidates']=candidates
        return data

    def noisify_params(self, model_params, anneal):
        """ Noisify model params.

        Noisify the given model parameters according to self.noise_policy
        and the annealing object provided. The noise_policy of some model
        parameter PARAM will only be applied if the annealing object
        provides a noise strength via PARAM_noise.

        """
        #H, D = self.H, self.D
        normal = np.random.normal
        uniform = np.random.uniform
        comm = self.comm

        for param, policy in list(self.noise_policy.items()):
            pvalue = model_params[param]
            if (not param+'_noise'=='pi_noise') and anneal[param+"_noise"] != 0.0:
                if np.isscalar(pvalue):         # Param to be noisified is scalar
                    new_pvalue = 0
                    if comm.rank == 0:
                        scale = anneal[param+"_noise"]
                        new_pvalue = pvalue + normal(scale=scale)
                        if new_pvalue < policy[0]:
                            new_pvalue = policy[0]
                        if new_pvalue >= policy[1]:
                            new_pvalue = policy[1]
                        if policy[2]:
                            new_pvalue = np.abs(new_pvalue)
                    pvalue = comm.bcast(new_pvalue)
                else:                                  # Param to be noisified is an ndarray
                    if comm.rank == 0:
                        scale = anneal[param+"_noise"]
                        shape = pvalue.shape
                        new_pvalue = pvalue + normal(scale=scale, size=shape)
                        low_bound, up_bound, absify = policy
                        new_pvalue = np.maximum(low_bound, new_pvalue)
                        new_pvalue = np.minimum( up_bound, new_pvalue)
                        if absify:
                            new_pvalue = np.abs(new_pvalue)
                        pvalue = new_pvalue
                    comm.Bcast([pvalue, MPI.DOUBLE])
            elif param+'_noise'=='pi_noise' and anneal["pi_noise"] != 0.0:
                if comm.rank == 0:
                    scale = anneal[param+"_noise"]
                    shape = pvalue.shape
                    new_pvalue = pvalue + np.random.rand(*shape)*scale
                    new_pvalue = new_pvalue/new_pvalue.sum()
                    pvalue  =   new_pvalue
                comm.Bcast([pvalue, MPI.DOUBLE])

            model_params[param] = pvalue

        return model_params



    def E_step(self, anneal, model_params, my_data):
        """ LinCA E_step

        my_data variables used:

            my_data['y']           Datapoints
            my_data['can']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """
        my_N, D =   my_data['y'].shape
        H       =   self.H
        SM      =   self.SM
        pp("E_step\n")
        W       =   model_params['W'].T
        pi      =   model_params['pi']
        sigma   =   model_params['sigma']
        states  =   self.states
        # Precompute
        beta    =   1./anneal['T']
        pre1    =   -1./2./sigma/sigma
        l       =   len(states)

        l_pis=np.zeros((self.no_states))
        for i in range(l):
            l_pis+=self.state_abs[i]*np.log(pi[i])
        # Allocate return structures
        F       =   np.empty( [my_N, 1 + (self.K-1)*H + self.no_states] )
        pre_F   =   np.empty( [1 + (self.K-1)*H + self.no_states] )
        # Iterate over all datapoints

################ Identify Inference Latent vectors##############
###########################################################################
        pre_F[0]  =   self.H * np.log(pi[self.K_0])
        c=0
        for state in range(self.K):
            if state == self.K_0:
                continue
            pre_F[c*H+1:(c+1)*H+1]  =   np.log(pi[state]) + ((self.H-1)*np.log(pi[self.K_0]))
            c+=1
        pre_F[(self.K-1)*H+1:]  =   l_pis
        pp("E_step: Before data loop")
        for n in range(my_N):
            y    = my_data['y'][n,:]
            cand = my_data['candidates'][n,:]
            # print "cand  ", cand

            # Handle hidden states with zero active hidden causes
            log_prod_joint = pre1 * (y**2).sum()
            F[n,0] = log_prod_joint

            # Handle hidden states with 1 active cause
            # import ipdb;ipdb.set_trace()
            log_prod_joint = pre1 * ((np.dot(self.SSM,W)-y)**2).sum(axis=1)
            F[n,1:(self.K-1)*H+1] = log_prod_joint

            if self.gamma>1:
                # Handle hidden states with more than 1 active cause
                W_   = W[cand]                          # is (Hprime x D)

                Wbar = np.dot(SM,W_)
                log_prod_joint = pre1 * (((Wbar-y)**2).sum(axis=1))
                F[n,(self.K-1)*H+1:] = log_prod_joint#+l_pis

        pp("Iteration anneal: {} ".format(anneal.cur_pos))
        if anneal['anneal_prior']:
            F[:,:] += pre_F[None,:]
            F[:,:] *= beta
            pp( "anneal prior with beta = {}".format(beta))
        else:
            F[:,:] *=beta
            F[:,:] += pre_F[None,:]
            pp( "not anneal prior with beta = {}".format(beta))
        return { 'logpj': F}#, 'denoms': denoms}

    def M_step(self, anneal, model_params, my_suff_stat, my_data):
        """ LinCA M_step

        my_data variables used:

            my_data['y']           Datapoints
            my_data['candidates']         Candidate H's according to selection func.

        Annealing variables used:

            anneal['T']            Temperature for det. annealing
            anneal['N_cut_factor'] 0.: no truncation; 1. trunc. according to model

        """
        comm      = self.comm
        W         = model_params['W'].T
        pi        = model_params['pi']
        sigma     = model_params['sigma']

        # Read in data:
        my_y       = my_data['y'].copy()
        candidates = my_data['candidates']
        logpj_all  = my_suff_stat['logpj']
        all_denoms = np.exp(logpj_all).sum(axis=1)
        my_N, D    = my_y.shape
        N          = comm.allreduce(my_N)

        A_pi_gamma = self.get_scaling_factors(model_params['pi'])
        dlog.append("prior_mass",A_pi_gamma)
        # _, A_pi_gamma, _=self.get_scaling_factors(model_params['pi'])

        #Truncate data
        N_use, my_y,candidates,logpj_all = self._get_sorted_data(N, anneal, A_pi_gamma, all_denoms, candidates,logpj_all,my_y)
        my_N, D = my_y.shape # update my_N

        # Precompute
        corr_all  = logpj_all.max(axis=1)                 # shape: (my_N,)
        pjb_all   = np.exp(logpj_all - corr_all[:, None])  # shape: (my_N, no_states)

        #Log-Likelihood:
        L = self.get_likelihood(D,sigma,A_pi_gamma,logpj_all,N_use)
        dlog.append('L',L)
        # Allocate
        my_Wp     = np.zeros_like(W)   # shape (H, D)
        my_Wq     = np.zeros((self.H,self.H))    # shape (H, H)
        my_pi     = np.zeros_like(pi)  # shape (K)
        my_sigma  = 0.0             #
        SM = self.SM

        # Iterate over all datapoints
        for n in range(my_N):
            y = my_y[n, :]                  # length D
            cand = candidates[n, :]  # length Hprime
            pjb = pjb_all[n, :]
            this_Wp = np.zeros_like(my_Wp)    # numerator for current datapoint (H, D)
            this_Wq = np.zeros_like(my_Wq)    # denominator for current datapoint (H, H)
            this_pi = np.zeros_like(pi)       # numerator for pi update (current datapoint)

            # Handle hidden states with 0 active causes
            this_pi[self.K_0] = self.H*pjb[0]
            this_sigma = pjb[0] * (y**2).sum()

            # Handle hidden states with 1 active cause
            #FIX: I am sure I need to multiply with pi somewhere here
            c=0
            # import ipdb;ipdb.set_trace()
            for state in range(self.K):
                if state == self.K_0:
                    continue
                sspjb = pjb[c*self.H+1:(c+1)*self.H+1]
                # this_Wp  += np.outer(sspjb,y.T)
                # this_Wq  += sspjb[:,None] * self.SSM[c*self.H:(c+1)*self.H]
                
                this_pi[state]  += sspjb.sum()

                recons = self.states[state]*W
                sqe = ((recons-y)**2).sum(1)
                this_sigma += (sspjb * sqe).sum()

                c+=1
            this_pi[self.K_0]  += ((self.H-1) * pjb[1:(self.K-1)*self.H+1]).sum()
            this_Wp         += np.dot(np.outer(y,pjb[1:(self.K-1)*self.H+1]),self.SSM).T
            # this_Wq_tmp           = np.zeros_like(my_Wq[cand])
            # this_Wq_tmp[:,cand]   = np.dot(pjb[(self.K-1)*self.H+1:] * SM.T,SM)
            this_Wq         += np.dot(pjb[1:(self.K-1)*self.H+1] * self.SSM.T, self.SSM)



            if self.gamma>1:
                # Handle hidden states with more than 1 active cause
                this_Wp[cand]         += np.dot(np.outer(y,pjb[(self.K-1)*self.H+1:]),SM).T
                this_Wq_tmp           = np.zeros_like(my_Wq[cand])
                this_Wq_tmp[:,cand]   = np.dot(pjb[(self.K-1)*self.H+1:] * SM.T,SM)
                this_Wq[cand]         += this_Wq_tmp

                this_pi += np.inner(pjb[(self.K-1)*self.H+1:], self.state_abs)

                W_ = W[cand]                           # is (Hprime x D)
                Wbar = np.dot(SM,W_)
                this_sigma += (pjb[(self.K-1)*self.H+1:] * ((Wbar-y)**2).sum(axis=1)).sum()
            #Scale down
            denom = pjb.sum()
            my_Wp += this_Wp / denom
            my_Wq += this_Wq / denom

            my_pi += this_pi / denom

            my_sigma += this_sigma/ denom/D

        #Calculate updated W
        Wp = np.empty_like(my_Wp)
        Wq = np.empty_like(my_Wq)
        comm.Allreduce( [my_Wp, MPI.DOUBLE], [Wp, MPI.DOUBLE] )
        comm.Allreduce( [my_Wq, MPI.DOUBLE], [Wq, MPI.DOUBLE] )
        # W_new  = np.dot(np.linalg.pinv(Wq), Wp)
        W_new  = np.linalg.lstsq(Wq, Wp)[0]    # TODO check and switch to this one

        # Calculate updated pi
        pi_new=np.empty_like(pi)
        # pi_new = E_pi_gamma * comm.allreduce(my_pi) / H / N_use
        for i in range(self.K):
            pi_new[i]  = comm.allreduce(my_pi[i])/comm.allreduce(my_pi.sum())

        eps = 1e-6
        if np.any(pi_new<eps):
            which_lo = pi_new<eps
            which_hi = pi_new>=eps
            pi_new[which_lo] += eps - pi_new[which_lo]
            pi_new[which_hi] -= (eps*np.sum(which_lo))/np.sum(which_hi)

        if 'penalty' in list(self.__dict__.keys()):
            self.penalty
            if self.penalty>pi_new[self.K_0]:
                r = (1-self.penalty)/(1-pi_new[self.K_0])
                pi_new[pi_new!=0] = pi_new[pi_new!=0]*r
                pi_new[self.K_0] = self.penalty
                pi_new/=pi_new.sum()

        # Calculate updated sigma
        sigma_new = np.sqrt(comm.allreduce(my_sigma) /  N_use)

        if 'W' not in self.to_learn:
            W_new = W
            pp("not learning W")
        if 'pi' not in self.to_learn:
            pi_new = pi
            pp("not learning pi")
        if 'sigma' not in self.to_learn:
            pp("not learning sigma")
            sigma_new = sigma

        for param in anneal.crit_params:
            exec('this_param = ' + param)
            anneal.dyn_param(param, this_param)

        dlog.append('N_use', N_use)

        return { 'W': W_new.transpose(), 'pi': pi_new, 'sigma': sigma_new, 'Q': 0.}

    def calculate_respons(self, anneal, model_params, data):
        data['candidates'].sort(axis=1) #(we do this to set the order back=outside)
        F_JB = self.E_step(anneal, model_params, data)['logpj']
        #Transform into responsibilities

        corr = np.max(F_JB, axis=1)
        exp_F_JB_corr = np.exp(F_JB - corr[:, None])
        respons = exp_F_JB_corr/(np.sum(exp_F_JB_corr, axis=1).reshape(-1, 1))
        return respons

    def free_energy(self, model_params, my_data):

        return 0.0

    def gain(self, old_parameters, new_parameters):

        return 0.0

    def get_scaling_factors(self,pi):
                # Precompute factor for pi update
        A_pi_gamma = 0.0
        # B_pi_gamma = np.zeros_like(pi)

        ar_gamma=np.arange(self.gamma+1)
        # iterator over all possible values of gamma_{k}
        p=itls.product(ar_gamma,repeat=len(self.states)-1)

        for gp in p:
            # gp: (tuple) holds (g_1,g_2,...,g_k)
            ngp=np.array(gp)
            if ngp.sum()>self.gamma:
                continue
            num0 = self.H-ngp.sum() #number of zeros for combinations gp, i.e. H-g_1-g_2-...-g_k
            abs_array = np.insert(ngp,self.K_0, num0)
            if not abs_array.sum() == self.H :
                raise Exception("wrong number of elements counted")
            cmb = multinom2(abs_array.sum(),abs_array)
            pm = np.prod(pi**abs_array)
            # pm2 = np.exp(np.sum(np.log(pi)*abs_array))
            # pp("cmb: {} \t for abs_array: {} \t and pm: {} \t and pm2: {} \t and pi: {}".format(cmb,abs_array,pm,pm2,pi))
            A_pi_gamma += cmb * pm
            # B_pi_gamma += np.prod(abs_array * cmb * (pi**abs_array))
        # E_pi_gamma = pi * self.H * A_pi_gamma / B_pi_gamma
        # return E_pi_gamma, A_pi_gamma, B_pi_gamma
        return A_pi_gamma

    def _get_sorted_data(self,N, anneal, A_pi_gamma, all_denoms, candidates,logpj_all,my_y):
        comm = self.comm
        pp("prior mass in _get_sorter_data: {}".format(A_pi_gamma))
        # if False:
        pp("N {} Ncut_factor {}".format(N,anneal['Ncut_factor']))
        if anneal['Ncut_factor'] > 0.0:

            N_use = int(N * (1 - (1 - A_pi_gamma) * anneal['Ncut_factor']))
            pp("N_use {} -> with cutting".format(N_use))
            cut_denom = parallel.allsort(all_denoms)[-N_use]
            pp("cut_denom {} ".format(cut_denom))
            which   = np.array(all_denoms > cut_denom)
            # which   = np.array(all_denoms >= cut_denom)
            candidates = candidates[which]
            logpj_all = logpj_all[which]
            my_y    = my_y[which]
            my_N, D = my_y.shape
            N_use = comm.allreduce(my_N)
            # N_use = N
            pp("N_use {} -> with cutting".format(N_use))
        else:
            N_use = N
            pp("N_use {} -> no cutting".format(N_use))

        pp("N_use {} Ncut_factor {}".format(N_use,anneal['Ncut_factor']))
        return N_use, my_y,candidates,logpj_all

    def get_likelihood(self,D,sigma,A,logpj_all,N):
        comm = self.comm
        L =  - 0.5 * D * np.log(2*np.pi*sigma**2)#-np.log(A)
        Fs = logsumexp(logpj_all,1).sum()
        L += comm.allreduce(Fs)/N
        return L
