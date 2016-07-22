#!/usr/bin/env python

import sys
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
# corrupt test data for inference

# Number of training datapoints
N = train_data.shape[0]
# Diemnsionality of the model
D = train_data.shape[1]
H = 600
# H = int(D*1.5)     # number of latents
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
    mean_factors = np.mean(my_y,0)
    scaling_factors = np.std(my_y,0)
    nzsf = scaling_factors!=0.
    my_y -= mean_factors[None,:]
    my_y /= scaling_factors[None,:]

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
    dlog.set_handler(('W', 'pi', 'sigma', 'y', 'ty','ts', 'N','scaling_factors','mean_factors','L','train_resp','test_resp',
                        'train_recon','test_recon','train_post','test_post','test_data_corrupted','test_data'), StoreToH5, output_path +'/result.h5')

    dlog.append('test_data',test_data)
    test_data[:,-160:] = 0.
    dlog.append('test_data_corrupted',test_data)
    dlog.append('y',my_y)
    dlog.append('scaling_factors',scaling_factors)
    dlog.append('mean_factors',mean_factors)
    dlog.append('N',N)

    model_params ={}
    model_params = model.standard_init(my_data)
    # pprint(model_params)
    # Choose annealing schedule
    anneal = LinearAnnealing(200)
    anneal['T'] = [(0., 1.25), (.25, 1.)]
    anneal['Ncut_factor'] = [(0, .5), (.2, 1.)]
    anneal['W_noise'] = [(0, 2.0), (.25, 0.0)]
    anneal['anneal_prior'] = False

    # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    em.run()

    # Infer test and some train_dara
    last_params = em.lparams
    my_train_data = {}
    my_train_data['y'] = train_data
    my_test_data = {}
    my_test_data['y'] = test_data
    train_responses = model.inference(anneal, last_params, my_train_data)
    test_responses = model.inference(anneal, last_params, my_test_data)
    train_recon = np.dot(train_responses['s'][:,0,:],last_params['W'].T)*scaling_factors[None,:] + mean_factors[None,:]
    test_recon = np.dot(test_responses['s'][:,0,:],last_params['W'].T)*scaling_factors[None,:] + mean_factors[None,:]

    dlog.append('train_resp', train_responses['s'][:,0,:])
    dlog.append('test_resp', test_responses['s'][:,0,:])
    dlog.append('train_post', train_responses['p'][:,:])
    dlog.append('test_post', test_responses['p'][:,:])
    dlog.append('train_recon', train_recon)
    dlog.append('test_recon', test_recon)


    dlog.close(True)
    pprint("Done")
