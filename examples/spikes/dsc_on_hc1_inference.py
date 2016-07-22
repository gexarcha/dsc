#!/usr/bin/env python

import sys
sys.path.insert(0, '../..')

from mpi4py import MPI
import numpy as np
np.random.RandomState(1023)

# from pulp.utils import create_output_path
from pulp.utils.parallel import pprint
# from pulp.utils.autotable import AutoTable
import tables
from pulp.utils.parallel import pprint, stride_data
from read_hc1 import readdat, readxml, getpatches, readfil
# from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
# from pulp.visualize.gui import GUI, RFViewer, YTPlotter

# from pulp.em import EM
from pulp.em.annealing import LinearAnnealing
# from pulp.utils.vc import VClog
import os

# output_path = create_output_path()
output_path = 'output/'+'dsc_on_hc1_run.py.2016-02-22+18:23/'
# run_tag = "run_"+output_path[output_path.find('.py.')+4:-1].replace(":","_")
# gitlog=VClog()
# if gitlog.repo.is_dirty():
#     gitlog.autocommit(MPI.COMM_WORLD,run_tag=run_tag)
# else:
#     if MPI.COMM_WORLD==0:
#         gitlog.tag(run_tag)

def rank0only(func, *args, **kargs):
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
# np.__config__.show()
# Number of datapoints to generate
fh = tables.open_file(output_path+'result.h5','a')
N = fh.root.N.read().squeeze()
W = fh.root.W[-1,:,:].squeeze()
pi = fh.root.pi[-1].squeeze()
sigma = fh.root.sigma[-1].squeeze()
gamma = fh.root.gamma.read().squeeze()
Hprime = fh.root.Hprime.read().squeeze()
states = fh.root.states.read().squeeze()
# Each datapoint is of D = size*size
# size = 16

# Diemnsionality of the model
D = W.shape[0]    # dimensionality of observed data
H = W.shape[1]
overlap = .5
# Approximation parameters for Expectation Truncation
# Hprime = 6
# gamma = 4

# Define Desired States
# states=np.array([-1.,0.,1.])
# states=np.array([0.,1.])
# states=np.array([0.,1.,2.,3.,4.])
# states=np.array([0.,1.,2.,3.,4.])

# Import and instantiate a model
from pulp.em.dmodels.dsc_et import DSC_ET
model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['W','pi','sigma'])
# model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['sigma','W'])

mparams = {'W':W,'sigma':sigma,'pi':pi}
# all_data = np.zeros(N,D)
nprocs  = MPI.COMM_WORLD.size
data=None
if MPI.COMM_WORLD.rank==0:
    fname = '/users/ml/xoex6879/data/crcns/hc-1/Data/d5611/d561102'
    xml = readxml(fname)
    data = readfil(fname+'F400_4000',xml['nChannels'])[1]
    mN = 10000
    all_data = getpatches(data,N,D,overlap)
    DC = getpatches(data,mN,D,.0, start=200000).mean()
    #DC = all_data.mean()

    all_data = all_data.reshape((nprocs,N/nprocs,D)) - DC
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

    all_data = comm.scatter(all_data,root=0)

    # Extract some parameters

    # Load data
    # data_fname = "/users/ml/xoex6879/data/VanHaterenImages/natims_conv_1700_p26.h5"
    # data_fname = "/users/ml/xoex6879/data/VanHaterenImages/natims_conv_1700_p16_95var.h5"
    # with tables.open_file(data_fname, 'r') as data_h5:
        # N_file = data_h5.root.wdata.shape[0]
        # if N_file < N:
        #     dlog.progress(
        #         "WARNING: N={} chosen but only {} data points available. ".format(N, N_file))
        #     N = N_file

        # first_y, last_y = stride_data(N)
        # my_y = data_h5.root.wdata[first_y:last_y]
        # data_h5.close()

    my_data = {
        'y': all_data,
    }
    print MPI.COMM_WORLD.rank, all_data.shape
    # pprint("{}".format(my_y[8,:10]))
    # Configure DataLogger
    print_list = ('T', 'Q', 'pi', 'sigma', 'N','N_use', 'L','W_noise','sigma_noise','pi_noise','prior_mass')
    h5_list = ('W', 'pi', 'sigma', 'y', 'N',  'N_use','prior_mass','states','Hprime','H','gamma')
    h5_list += ('infered_posterior','infered_states','series','rseries','ry','rs')
    # dlog.set_handler(['L'], YTPlotter)
    # dlog.set_handler(print_list, TextPrinter)
    # dlog.set_handler(print_list, StoreToTxt, output_path + '/terminal.txt')
    # dlog.set_handler(h5_list, StoreToH5, output_path +'/result.h5')
    dlog.('overlap',overlap)


    # Choose annealing schedule
    anneal = LinearAnnealing(200)
    anneal['T'] = [(0., 1.)]
    # anneal['T'] = [(0.0, 2.), (0.05,2.), (0.4, 1.)]
    # anneal['W_noise'] = [(0.0, 0.), (0.05,0.), (0.15,1.), (0.3,3.), (0.5, 0.)]
    # anneal['W_noise'] = [(0.0, 0.2), (0.05,0.2), (0.5, 0.)]
    # anneal['W_noise'] = [(0.0, 1.),(0.1, 1./10.0), (0.5, 0.0)]
    # anneal['T'] = [(0.0, 1.),(0.1,1.), (0.5, 1.)]
    anneal['Ncut_factor'] = [(0.0, 1.0)]
    # anneal['Ncut_factor'] = [(0.0, 0.1),(0.5, 1.0)]
    # anneal['W_noise'] = [(0.0, 1./10),(0.1, 1./10.0), (0.5, 0.0)]
    anneal['anneal_prior'] = False

    # Create and start EM annealing
    # em = EM(model=model, anneal=anneal)
    # em.data = my_data
    # em.lparams = model_params
    # em.run()

    res = model.inference(anneal, mparams, {'y':all_data})
    inf_poster = np.zeros((res['p'].shape[0]*comm.size,)+res['p'].shape[1:])
    inf_states = np.zeros((res['s'].shape[0]*comm.size,)+res['s'].shape[1:],dtype='int64')
    # comm.Gather
    comm.Allgather((res['p'],MPI.DOUBLE),(inf_poster,MPI.DOUBLE))
    comm.Allgather((res['s'],MPI.INT),(inf_states,MPI.INT))
    # dlog.append('infered_posterior',inf_poster)
    # dlog.append('infered_states',inf_states)
    if  comm.rank==0:
        ipo = tables.EArray( fh.root, 'infered_posterior',tables.Float64Atom(),(0,)+inf_poster.shape[1:])
        ipo.append(inf_poster)
        ist = tables.EArray( fh.root, 'infered_states',tables.Int8Atom(),(0,)+inf_states.shape[1:])
        ist.append(inf_states)
        recon = model.generate_data(mparams, all_data.shape[0]*comm.size, noise_on=False, gs=inf_states[:,0,:])
        ry = tables.EArray( fh.root, 'ry',tables.Float64Atom(),(0,)+recon['y'].shape[1:])
        ry.append(recon['y'])
        rs = tables.EArray( fh.root, 'rs',tables.Float64Atom(),(0,)+recon['s'].shape[1:])
        rs.append(recon['s'])
        s=0
        step=D*(1-overlap)
        reconseries = np.zeros((N*step + D*overlap,))
        for n in range(N):
            if n==0:
                # s = n*D
                e = s+D
                reconseries[s:e]=recon['y'][n]
            else:
                if inf_poster[n,0]>inf_poster[n-1,0]:
                    # s=n*D*(1-overlap)
                    e=s+D
                    # print n,s,e
                    reconseries[s:e]=recon['y'][n]
                else:
                    # s=n*D*(1-overlap)
                    # s+=
                    e=s+D
                    # print n,s,e
                    reconseries[s+np.ceil(overlap*D):e]=recon['y'][n,np.ceil(overlap*D):D]
            s+=step
        ry = tables.EArray( fh.root, 'rseries',tables.Float64Atom(),(0,)+reconseries.shape[1:])
        ry.append(reconseries)
        rs = tables.EArray( fh.root, 'series',tables.Float64Atom(),(0,)+data.shape[1:])
        rs.append(data)

    # dlog.close(True)
    pprint("Done")
fh.close()
