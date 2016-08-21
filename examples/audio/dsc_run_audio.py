#!/usr/bin/env python
#
#  Lincense: Academic Free License (AFL) v3.0
#


import sys
sys.path.insert(0, '../..')

from mpi4py import MPI
import numpy as np
np.random.RandomState(1023)

from dsc.utils import create_output_path
from dsc.utils.parallel import pprint
# from dsc.utils.barstest import generate_bars
# from dsc.utils.autotable import AutoTable
import tables
from dsc.utils.parallel import pprint, stride_data

from dsc.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
# from dsc.visualize.gui import GUI, RFViewer, YTPlotter

from dsc.em import EM
from dsc.em.annealing import LinearAnnealing

from data import read
import os

output_path = create_output_path()

def rank0only(func, *args, **kargs):
    """performs function on node 0 only

    
    It will call function func with arguments args and keyword
    arguments kargs on node 0 only
    
    Arguments:
        func {python function} -- the function to be called by the root node
        *args {list} -- tuple or list that contains the arguments to be passed 
                        to the function
        **kargs {dictionary} -- dictionary with keys the keyword arguments
    """
    if MPI.COMM_WORLD.rank == 0:
        func(*args, **kargs)
    MPI.COMM_WORLD.barrier()


if MPI.COMM_WORLD.rank==0:
    exclude_from_sync = ['output','*.err','*.out','*jpg','*png','*gif','.nfs*','*/.*','*pyc','*pyx']
    cmd = "rsync -avz "
    for pattern in exclude_from_sync:
        cmd = cmd + "--exclude '{}' ".format(pattern)
    # cmd = cmd + " {} {}".format('/'+''.join(output_path.split('/')[:-1]),os.path.abspath('../..'), output_path)
    cmd = cmd + " {} {}".format(os.path.abspath('../..'), output_path+'src')
    print "Executing :{}".format(cmd)
    os.system(cmd)
# Number of datapoints to generate
N = 200000


# Each datapoint is of D = size*size
# size = 3
overlap=0.5

# Diemnsionality of the model
H = 200     # number of latents
D = 160
# Approximation parameters for Expectation Truncation
Hprime = 6
gamma = 4
states=np.array([-2.,-1.,0.,1.,2.])

# Import and instantiate a model
from dsc.models.dsc_et import DSC_ET
model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['W','sigma','pi'])



# Read Data

nprocs  = MPI.COMM_WORLD.size
data=None
channel = 1
if MPI.COMM_WORLD.rank==0:
    data = read(N,D,overlap)


    N=data.shape[0]
    N = N- (N%nprocs)
    data=data[:N,:]
    all_data = data.reshape((nprocs,N/nprocs,D)) 
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

    # Load data
    data_fname = "/users/ml/xoex6879/data/cat/h5" + "/tiger_p6_highpass_high.h5"
    my_y=all_data
    my_data = {
        'y': my_y,
    }

    # Prepare model...

    # Configure DataLogger
    print_list = ('T', 'Q', 'pi', 'sigma', 'N','N_use', 'L','W_noise','sigma_noise','pi_noise','prior_mass')
    h5_list = ('W', 'pi', 'sigma', 'y', 'N',  'N_use','prior_mass','states','Hprime','H','gamma','channel')
    h5_list += ('infered_posterior','infered_states','series','rseries','ry','rs','overlap','ty','ts','overlap')
    dlog.set_handler(print_list, TextPrinter)
    dlog.set_handler( h5_list, StoreToH5, output_path +'/result.h5')
    dlog.append('y',my_y)
    dlog.append('N',N)
    dlog.append('gamma',gamma)
    dlog.append('H',H)
    dlog.append('overlap',overlap)
    dlog.append('states',states)

    model_params ={}

    model_params = model.standard_init(my_data)
    #Initialize W with data points
    model_params['W'][:,:] = my_y[np.abs(my_y).sum(1)>0.5][:H,:].T 
    comm.Bcast([model_params['W'], MPI.DOUBLE])

    # Choose annealing schedule
    anneal = LinearAnnealing(200)
    anneal['T'] = [(0., 10.), (.25, 1.)]
    anneal['Ncut_factor'] = [(0,0.),(0.1,0.), (.4, 1.)]
    anneal['W_noise'] = [(0, .1), (0.1, .1), (.25, 0.0)]
    anneal['anneal_prior'] = False





    # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    em.run()
    #Inference
    res = model.inference(anneal, em.lparams, {'y':my_y})
    inf_poster = np.zeros((res['p'].shape[0]*comm.size,)+res['p'].shape[1:])
    inf_states = np.zeros((res['s'].shape[0]*comm.size,)+res['s'].shape[1:],dtype='int64')
    # comm.Gather
    comm.Allgather((res['p'],MPI.DOUBLE),(inf_poster,MPI.DOUBLE))
    comm.Allgather((res['s'],MPI.INT),(inf_states,MPI.INT))
    dlog.append('infered_posterior',inf_poster)
    dlog.append('infered_states',inf_states)

    if  comm.rank==0:
        recon = model.generate_data(em.lparams, my_y.shape[0]*comm.size, noise_on=False, gs=inf_states[:,0,:])
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
