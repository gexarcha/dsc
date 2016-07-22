#!/usr/bin/env python

import sys
sys.path.insert(0, '..')

from mpi4py import MPI
import numpy as np
np.random.RandomState(1023)

from pulp.utils import create_output_path
from pulp.utils.parallel import pprint
# from pulp.utils.barstest import generate_bars
# from pulp.utils.autotable import AutoTable
import tables
from pulp.utils.parallel import pprint, stride_data

from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
# from pulp.visualize.gui import GUI, RFViewer, YTPlotter

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing

output_path = create_output_path()

def rank0only(func, *args, **kargs):
    if MPI.COMM_WORLD.rank == 0:
        func(*args, **kargs)
    MPI.COMM_WORLD.barrier()


# Number of datapoints to generate
N = 20000

# Each datapoint is of D = size*size
size = 3

# Diemnsionality of the model
H = 35     # number of latents
D = 30
# Approximation parameters for Expectation Truncation
Hprime = 6
gamma = 4
# states=np.array([-1.,0.,1.])
# states=np.array([0.,1.,2.,3.])
# states=np.array([-2.,-1.,0.,1.,2.])
states=np.array([0.,1.,2.,3.,4.,5.])
# states=np.array([0.,1.])
# states=np.array([0.,1.,2.])

# Import and instantiate a model
from pulp.em.dmodels.dsc_et import DSC_ET
model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['W','sigma','pi'])
# model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['W','pi'])
# model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['sigma','W'])

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
    data_fname = "/users/ml/xoex6879/data/cat/h5" + "/tiger_p6_highpass_high.h5"
    channel = 25
    # data_fname = output_path + "/data.h5"
    with tables.openFile(data_fname, 'r') as data_h5:
        # N_file = 
        vp=data_h5.root.data.patches.neuron_25.read()
        first_y, last_y = stride_data(N)
        my_y = vp[:,first_y:last_y]
        del vp
        # my_s = data_h5.root.s[first_y:last_y]
        data_h5.close()

    my_data = {
        'y': my_y.T,
        # 's': my_s,
    }

    # Prepare model...

    # Configure DataLogger
    # dlog.start_gui(GUI)
    print_list = ('T', 'Q', 'pi', 'sigma', 'N', 'MAE', 'L')
    # dlog.set_handler(['L'], YTPlotter)
    dlog.set_handler(print_list, TextPrinter)
    #dlog.set_handler(print_list, StoreToTxt, output_path + '/terminal.txt')
    #dlog.set_handler('Q', YTPlotter)
    # dlog.set_handler('W', RFViewer, rf_shape=(D2, D2))
    # dlog.set_handler('W_gt', RFViewer, rf_shape=(D2, D2))
    # dlog.append('W_gt',params_gt['W'])
    dlog.set_handler(('W', 'pi', 'sigma', 'mu', 'y', 'MAE', 'N'), StoreToH5, output_path +'/result.h5')
    # if 'pi' in model.to_learn:
        # dlog.set_handler(['pi'], YTPlotter)
    # if 'pies' in model.to_learn:
    #     dlog.set_handler(['pies'], YTPlotter)
    # if 'sigma' in model.to_learn:
    #     dlog.set_handler(['sigma'], YTPlotter)
    # if 'mu' in model.to_learn:
    #     dlog.set_handler(['mu'], YTPlotter)
    # dlog.set_handler('y', RFViewer, rf_shape=(D2, D2))
    # dlog.append('y',my_y[:10].T)

    model_params = model.standard_init(my_data)
    # for param_name in model_params.keys():
    #     if param_name not in model.to_learn:
    #         model_params[param_name]=params_gt[param_name]

    # model.s_gt = my_s
    # Choose annealing schedule
    anneal = LinearAnnealing(100)
    anneal['T'] = [(0., 10.), (.25, 1.)]
    anneal['Ncut_factor'] = [(0, .5), (.2, 1.)]
    anneal['W_noise'] = [(0, 2.0), (.25, 0.0)]
    anneal['anneal_prior'] = False

    # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    em.run()

    dlog.close(True)
    pprint("Done")