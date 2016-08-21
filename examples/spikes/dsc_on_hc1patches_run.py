#!/usr/bin/env python
#
#  Lincense: Academic Free License (AFL) v3.0
#

import sys
sys.path.insert(0, '../..')

from mpi4py import MPI
import numpy as np
np.random.RandomState(1023)

from pulp.utils import create_output_path
from pulp.utils.parallel import pprint
from read_hc1 import  readxml, getpatches, readfil
from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing
import os

output_path = create_output_path()

def rank0only(func, *args, **kargs):
    if MPI.COMM_WORLD.rank == 0:
        func(*args, **kargs)
    MPI.COMM_WORLD.barrier()

if MPI.COMM_WORLD.rank==0:
    exclude_from_sync = ['output','*.err','*.out','*jpg','*png','*gif','.nfs*','*/.*','*pyc','*pyx']
    cmd = "rsync -avz "
    for pattern in exclude_from_sync:
        cmd = cmd + "--exclude '{}' ".format(pattern)
    cmd = cmd + " {} {}".format(os.path.abspath('../..'), output_path+'src')
    print "Executing :{}".format(cmd)
    os.system(cmd)
# Number of datapoints to generate
N = 100000


# Diemnsionality of the model
D = 20    # dimensionality of observed data
H = 5
overlap = 0.0
# Approximation parameters for Expectation Truncation
Hprime = 5
gamma = 3

# Define Desired States
# states=np.array([0.,1.,2.,3.,4.])
# states=np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.])
states=np.array([0.,1.,15,20])
# states=np.array([0.,1.,5.])

# Import and instantiate a model
from pulp.em.dmodels.dsc_et import DSC_ET
model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['W','pi','sigma'])
nprocs  = MPI.COMM_WORLD.size
data=None
channel = 1
if MPI.COMM_WORLD.rank==0:
    # fname = '/users/ml/xoex6879/data/crcns/hc-1/Data/d5611/d561102'
    fname = '/users/ml/xoex6879/data/crcns/hc-1/Data/d5331/d533101'
    xml = readxml(fname)
    data = readfil(fname+'F400_4000',xml['nChannels'])
    data -= data.mean(1)[:,None]
    mN = 10000
    all_data = getpatches(data[channel],N,D,overlap)
    N=all_data.shape[0]
    N = N- (N%nprocs)
    all_data=all_data[:N,:]
    all_data = all_data.reshape((nprocs,N/nprocs,D)) 
else:
    all_data = None

# Main
if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    pprint("=" * 40)
    pprint(" Running %d parallel processes" % comm.size)
    pprint("=" * 40)

    # Parse command line arguments TODO: bit of error checking
    if "output_path" not in globals():
        output_path = sys.argv[2]

    N = comm.bcast(N,root=0)
    all_data = comm.scatter(all_data,root=0)

    # Extract some parameters


    my_data = {
        'y': all_data,
    }
    # Configure DataLogger
    print_list = ('T', 'Q', 'pi', 'sigma', 'N','N_use', 'L','W_noise','sigma_noise','pi_noise','prior_mass')
    h5_list = ('W', 'pi', 'sigma', 'y', 'N',  'N_use','prior_mass','states','Hprime','H','gamma','channel')
    h5_list += ('infered_posterior','infered_states','series','rseries','ry','rs')
    dlog.set_handler(print_list, TextPrinter)
    dlog.set_handler(print_list, StoreToTxt, output_path + '/terminal.txt')
    dlog.set_handler(h5_list, StoreToH5, output_path +'/result.h5')
    dlog.append('N',N)
    dlog.append('Hprime',Hprime)
    dlog.append('H',H)
    dlog.append('gamma',gamma)
    dlog.append('states',states)
    dlog.append('channel',channel)
    dlog.append('overlap',overlap)
    # Prepare model...
    model_params = model.standard_init(my_data)
    dlog.append_all(model_params)

    # Choose annealing schedule
    anneal = LinearAnnealing(200)
    anneal['T'] = [(0.0, 2.), (0.05,2.), (0.4, 1.)]
    # anneal['W_noise'] = [(0.0, 1./10),(0.1, 1./10.0), (0.5, 0.0)]
    anneal['anneal_prior'] = False

    # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    em.run()

    res = model.inference(anneal, em.lparams, {'y':all_data})
    inf_poster = np.zeros((res['p'].shape[0]*comm.size,)+res['p'].shape[1:])
    inf_states = np.zeros((res['s'].shape[0]*comm.size,)+res['s'].shape[1:],dtype='int64')
    # comm.Gather
    comm.Allgather((res['p'],MPI.DOUBLE),(inf_poster,MPI.DOUBLE))
    comm.Allgather((res['s'],MPI.INT),(inf_states,MPI.INT))
    dlog.append('infered_posterior',inf_poster)
    dlog.append('infered_states',inf_states)
    if  comm.rank==0:
        recon = model.generate_data(em.lparams, all_data.shape[0]*comm.size, noise_on=False, gs=inf_states[:,0,:])
        dlog.append('ry',recon['y'])
        dlog.append('rs',recon['s'])
        s=0
        step=D*(1-overlap)
        reconseries = np.zeros((N*step + D*overlap,))
        for n in range(N):
            if n==0:
                e = s+D
                reconseries[s:e]=recon['y'][n]
            else:
                if inf_poster[n,0]>inf_poster[n-1,0]:
                    e=s+D
                    reconseries[s:e]=recon['y'][n]
                else:
                    e=s+D
                    reconseries[s+np.ceil(overlap*D):e]=recon['y'][n,np.ceil(overlap*D):D]
            s+=step
        dlog.append('rseries',reconseries)
        dlog.append('series',data)

    dlog.close(True)
    pprint("Done")
