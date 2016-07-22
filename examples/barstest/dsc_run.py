#!/usr/bin/env python

import sys
sys.path.insert(0, '../..')

from mpi4py import MPI
import numpy as np
np.random.RandomState(103)

from dsc.utils import create_output_path
from dsc.utils.parallel import pprint
from dsc.utils.barstest import generate_bars
from dsc.utils.autotable import AutoTable
import tables
from dsc.utils.parallel import pprint, stride_data

from dsc.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
# from dsc.visualize.gui import GUI, RFViewer, YTPlotter

from dsc.em import EM
from dsc.em.annealing import LinearAnnealing
# from dsc.utils.vc import VClog
import os 

output_path = create_output_path()

if MPI.COMM_WORLD.rank==0:
    exclude_from_sync = ['output','*.err','*.out','*jpg','*png','*gif','.nfs*','*/.*','*pyc','*pyx']
    cmd = "rsync -avz "
    for pattern in exclude_from_sync:
        cmd = cmd + "--exclude '{}' ".format(pattern)
    # cmd = cmd + " {} {}".format('/'+''.join(output_path.split('/')[:-1]),os.path.abspath('../..'), output_path)
    cmd = cmd + " {} {}".format(os.path.abspath('../..'), output_path+'src')
    print ("Executing :{}".format(cmd))
    os.system(cmd)

def rank0only(func, *args, **kargs):
    if MPI.COMM_WORLD.rank == 0:
        func(*args, **kargs)
    MPI.COMM_WORLD.barrier()


# Number of datapoints to generate
N = 1000

# Each datapoint is of D = size*size
size = 5

# Diemnsionality of the model
H = 2 * size     # number of latents
D = size ** 2    # dimensionality of observed data

# Approximation parameters for Expectation Truncation
Hprime = 7
gamma = 5
# states=np.array([-1.,0.,1.])
# states=np.array([0.,1.,2.,3.])
states=np.array([-2.,-1.,0.,1.,2.])
# states=np.array([0.,1.,2.,3.,4.,5.])
# states=np.array([-2.,0.,1.,5.])
# states=np.array([-2.,0.,1.])
# states=np.array([0.,1.,2.,3.,4.])
# states=np.array([0.,1.])
# states=np.array([0.,1.,2.])

# Import and instantiate a model
from dsc.models.dsc_et import DSC_ET
# model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['W','pi','sigma'])
model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['sigma','W','pi'])

# Model parameters used when artificially generating
# ground-truth data. This will NOT be used for the learning
# process.
# pi_gt = np.array([ 0.8, 0.2])
# pi_gt = np.array([ 0.15,0.8, 0.05])
# pi_gt = np.array([ 0.7, 0.2, 0.1])
# pi_gt = np.array([ 0.7, 0.2, 0.1])
pi_gt = np.array([ 0.1, 0.8, 0.1])
pi_gt = np.array([0.025, 0.075, 0.8, 0.075, 0.025])
# pi_gt = np.array([0.8,  0.05, 0.025, 0.1, 0.025])
# pi_gt = np.array([0.1, 0.7, 0.15, 0.05])

pi_gt = pi_gt/pi_gt.sum()
print (pi_gt)
params_gt = {
    'W':  10 * generate_bars(H),   # this function is in bars-create-data
    'pi':  pi_gt,
    'sigma':  2.
}


if MPI.COMM_WORLD.rank == 0:
    # create data
    data = model.generate_data(params_gt, N)

    # and save results
    out_fname = output_path + "/data.h5"
    with AutoTable(out_fname) as tbl:

        # Save ground-truth parameters
        for key in params_gt:
            tbl.append(key, params_gt[key])

        # Save generated data
        for key in data:
            tbl.append(key, data[key])
        tbl.close()
MPI.COMM_WORLD.barrier()
# print(dir(MPI.COMM_WORLD))
# output_path=MPI.COMM_WORLD.bcast(output_path)
# print output_path
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

    D2 = H // 2
    assert D2 ** 2 == D

    # Load data
    data_fname = output_path + "/data.h5"
    # data_fname = "output/dsc_run.py.2015-12-04+17:42/data.h5"
    with tables.open_file(data_fname, 'r') as data_h5:
        N_file = data_h5.root.y.shape[1]
        if N_file < N:
            dlog.progress(
                "WARNING: N={} chosen but only {} data points available. ".format(N, N_file))
            N = N_file

        first_y, last_y = stride_data(N)
        my_y = data_h5.root.y[0][first_y:last_y]
        my_s = data_h5.root.s[first_y:last_y]
        data_h5.close()

    my_data = {
        'y': my_y,
        's': my_s,
    }
    # import ipdb;ipdb.set_trace()

    # Prepare model...

    # Configure DataLogger
    # dlog.start_gui(GUI)
    print_list = ('T', 'Q', 'pi', 'sigma', 'N', 'N_use', 'MAE', 'L')
    h5_list = ('W', 'pi', 'sigma', 'y', 'N', 'N_use', 'prior_mass', 'states', 'Hprime', 'H', 'gamma', 'mu', 'MAE','W_gt','pi_gt','sigma_gt')
    # dlog.set_handler(['L'], YTPlotter)
    dlog.set_handler(print_list, TextPrinter)
    dlog.set_handler(h5_list, StoreToH5, output_path +'/result.h5')
    #dlog.set_handler(print_list, StoreToTxt, output_path + '/terminal.txt')
    #dlog.set_handler('Q', YTPlotter)
    # dlog.set_handler('W', RFViewer, rf_shape=(D2, D2))
    # dlog.set_handler('W_gt', RFViewer, rf_shape=(D2, D2))
    dlog.append('W_gt',params_gt['W'])
    dlog.append('sigma_gt',params_gt['sigma'])
    dlog.append('pi_gt',params_gt['pi'])
    # if 'pi' in model.to_learn:
        # dlog.set_handler(['pi'], YTPlotter)
    # if 'pies' in model.to_learn:
    #     dlog.set_handler(['pies'], YTPlotter)
    dlog.append('N',N)
    dlog.append('Hprime',Hprime)
    dlog.append('H',H)
    dlog.append('gamma',gamma)
    dlog.append('states',states)
    # if 'sigma' in model.to_learn:
        # dlog.set_handler(['sigma'], YTPlotter)
    # if 'mu' in model.to_learn:
    #     dlog.set_handler(['mu'], YTPlotter)
    # dlog.set_handler('y', RFViewer, rf_shape=(D2, D2))
    # dlog.append('y',my_y[:10].T)

    model_params = model.standard_init(my_data)
    # model_params = params_gt
    # model_params['W'] = np.random.normal(size=model_params['W'].shape)+20
    # model_params['pi'] = np.array([0.1,0.6,0.1,0.2])
    # model_params = params_gt
    dlog.append_all(model_params)
    for param_name in model_params.keys():
        if param_name not in model.to_learn:
            model_params[param_name]=params_gt[param_name]

    model.s_gt = my_s
    # Choose annealing schedule
    anneal = LinearAnnealing(100)
    anneal['T'] = [(0.0, 2.), (0.1, 2.), (0.4, 1.)]
    anneal['Ncut_factor'] = [(0, 0.0),(0.6,0.0),  (0.9, 1.)]
    # anneal['Ncut_factor'] = [(0, 0.0),(0.1,0.0), (0.4, 1.)]
    # anneal['W_noise'] = [(0, 5.0), (.6, 0.0)]
    # anneal['pi_noise'] = [(0, 0.1), (.6, 0.0)]
    anneal['anneal_prior'] = False

    # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    em.run()

    dlog.close(True)
    pprint("Done")
