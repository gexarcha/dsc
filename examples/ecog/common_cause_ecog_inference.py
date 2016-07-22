#!/usr/bin/env python

import sys, os
sys.path.insert(0, '../..')

from mpi4py import MPI
import numpy as np
np.random.RandomState(1023)

from pulp.utils import create_output_path
from pulp.utils.parallel import pprint, stride_data
from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
from pulp.em import EM
from pulp.em.annealing import LinearAnnealing
from pulp.em.camodels.dsc_et import DSC_ET

import tables as tb
import h5py

output_path = create_output_path()

def rank0only(func, *args, **kargs):
    if MPI.COMM_WORLD.rank == 0:
        func(*args, **kargs)
    MPI.COMM_WORLD.barrier()

#Read data
fh = h5py.File("/users/ml/xoex6879/data/ECoG/ec14/aEC14_B2.mat",'r')
ecog_data = fh[fh['/ecog/data'][0,0]].value
ecog_dataFs = fh['/ecog/dataFs'].value[0,0]
audio = np.squeeze(fh['/ecog/audio'].value)
audioFs = fh['/ecog/audioFs'].value[0,0]
f=fh[fh['/ecog/f'].value[0,0]].value
fh.close()


ecog_data_5_30hz = np.mean(ecog_data[:,1:4,:],1).T
audio = audio.reshape((ecog_data_5_30hz.shape[0], audioFs/ecog_dataFs))
data = np.concatenate([ecog_data_5_30hz,audio],1)

train_data = data[:-1000,:]
test_data = data[-1000:,:]

# Number of training datapoints
N = train_data.shape[0]
# Diemnsionality of the model
D = train_data.shape[1]
H = int(D*1.5)     # number of latents
# Approximation parameters for Expectation Truncation
Hprime = 10
gamma = 4
states = np.array([0.,1.])

# Main
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    pprint("=" * 40)
    pprint(" Running %d parallel processes" % comm.size)
    pprint("=" * 40)

    # Parse command line arguments TODO: bit of error checking
    if "output_path" not in globals():
        output_path = sys.argv[2]

    # Extract some parameters

    # Load data
    # first_y, last_y = stride_data(N)
    # my_y = read_rand_phone_chunks(N,D)[first_y:last_y]
    my_y = train_data
    # scaling_factors = np.std(my_y,1)
    # nzsf = scaling_factors!=0.
    # my_y[nzsf] /= scaling_factors[nzsf,None]
    print "Data shape {}".format(my_y.shape)

    my_data = {
        'y': my_y,
        # 's': my_s,
    }

    # Import and instantiate a model
    model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['W','sigma','pi'])

    # Configure DataLogger
    print_list = ('T', 'Q', 'pi', 'sigma', 'N', 'MAE', 'L')
    # dlog.set_handler(['L'], YTPlotter)
    dlog.set_handler(print_list, TextPrinter)
    dlog.set_handler(print_list, StoreToTxt, output_path + '/terminal.txt')
    # dlog.append('W_gt',params_gt['W'])
    dlog.set_handler(('W', 'pi', 'sigma', 'y', 'ty','ts', 'N','scaling_factors','L'), StoreToH5, output_path +'/result.h5')
    dlog.append('y',my_y)
    # dlog.append('scaling_factors',scaling_factors)
    dlog.append('N',N)

    model_params ={}
    model_params = model.standard_init(my_data)

    RUN = 'common_cause_ecog.py.2015-08-27+14:39'
    fname = os.getenv("HOME")+'/workspace/pylib/examples/ecog/output/'+RUN+'/result.h5'
    # print fname
    fh = tb.openFile(fname,'r')
    model_params['W'] = fh.root.W[-1,:,:]
    model_params['pi'] = fh.root.pi[-1,:]
    model_params['sigma'] = fh.root.sigma[-1]
    # fh.close()
    pprint(model_params)
    # Choose annealing schedule
    anneal = LinearAnnealing(1)
    anneal['T'] = [(0., 1.), (.25, 1.)]
    # anneal['Ncut_factor'] = [(0, .5), (.2, 1.)]
    # anneal['W_noise'] = [(0, 2.0), (.25, 0.0)]
    anneal['anneal_prior'] = False

    # Create and start EM annealing
    # em = EM(model=model, anneal=anneal)
    # em.data = my_data
    # em.lparams = model_params
    # em.run()
    # TODO: add inference for train and test_data

    my_train_data = my_data
    my_train_data['y'] = train_data[-1000:,:]
    my_test_data = {}
    my_test_data['y'] = test_data
    train_responses = model.inference(anneal, model_params, my_train_data)
    test_responses = model.inference(anneal, model_params, my_test_data)
    train_recon = np.dot(train_responses['s'][:,0,:],model_params['W'].T)
    test_recon = np.dot(test_responses['s'][:,0,:],model_params['W'].T)

    dlog.close(True)
    pprint("Done")
