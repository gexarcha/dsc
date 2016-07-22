#!/usr/bin/env python

import sys
sys.path.insert(0, '../..')

from mpi4py import MPI
import numpy as np
np.random.RandomState(1023)

from pulp.utils import create_output_path
from pulp.utils.parallel import pprint
from pulp.utils.barstest import generate_bars
from pulp.utils.autotable import AutoTable
import tables
from pulp.utils.parallel import pprint, stride_data

from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
from pulp.visualize.gui import GUI, RFViewer, YTPlotter

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing
from pulp.utils.vc import is_dirty, autocommit, tag

print is_dirty()

output_path = create_output_path()
run_tag = "run_"+output_path[output_path.find('.py.')+4:-1].replace(":","_")
if is_dirty():
    autocommit(MPI.COMM_WORLD,run_tag=run_tag)
else:
    if MPI.COMM_WORLD==0:
        tag(run_tag)

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
gamma = 6
# states=np.array([-1.,0.,1.])
# states=np.array([0.,1.,2.,3.])
# states=np.array([-2.,-1.,0.,1.,2.])
# states=np.array([0.,1.,2.,3.,4.,5.])
states=np.array([0.,1.])
# states=np.array([0.,1.,2.])

# Import and instantiate a model
from pulp.em.dmodels.dsc_et import DSC_ET
model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['W','pi','sigma'])
# model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['sigma','W'])

# Model parameters used when artificially generating
# ground-truth data. This will NOT be used for the learning
# process.
pi_gt = np.array([ 0.8, 0.2])
# pi_gt = np.array([ 0.7, 0.1, 0.2])
# pi_gt = np.array([ 0.7, 0.2, 0.1])
# pi_gt = np.array([ 0.1, 0.8, 0.1])
# pi_gt = np.array([0.8, 0.1, 0.05, 0.025, 0.0125, 0.0125])

pi_gt = pi_gt/pi_gt.sum()
params_gt = {
    'W':  10 * generate_bars(H),   # this function is in bars-create-data
    'pi':  pi_gt,
    'sigma':  2.
}
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
    my_y = np.random.normal(scale=2.,size=(N,D))
    my_data = {
        'y': my_y,
    }
    # import ipdb;ipdb.set_trace()

    # Prepare model...

    # Configure DataLogger
    # dlog.start_gui(GUI)
    print_list = ('T', 'Q', 'pi', 'sigma', 'N', 'N_use', 'MAE', 'L')
    dlog.set_handler(['L'], YTPlotter)
    dlog.set_handler(print_list, TextPrinter)
    #dlog.set_handler(print_list, StoreToTxt, output_path + '/terminal.txt')
    #dlog.set_handler('Q', YTPlotter)
    dlog.set_handler('W', RFViewer, rf_shape=(D2, D2))
    dlog.set_handler('W_gt', RFViewer, rf_shape=(D2, D2))
    dlog.append('W_gt',params_gt['W'])
    #dlog.set_handler(('W', 'pi', 'sigma', 'mu', 'y', 'MAE', 'N'), StoreToH5, output_path +'/result.h5')
    # if 'pi' in model.to_learn:
        # dlog.set_handler(['pi'], YTPlotter)
    # if 'pies' in model.to_learn:
    #     dlog.set_handler(['pies'], YTPlotter)
    if 'sigma' in model.to_learn:
        dlog.set_handler(['sigma'], YTPlotter)
    # if 'mu' in model.to_learn:
    #     dlog.set_handler(['mu'], YTPlotter)
    # dlog.set_handler('y', RFViewer, rf_shape=(D2, D2))
    # dlog.append('y',my_y[:10].T)

    model_params = model.standard_init(my_data)
    # model_params = params_gt
    dlog.append_all(model_params)
    for param_name in model_params.keys():
        if param_name not in model.to_learn:
            model_params[param_name]=params_gt[param_name]

    # Choose annealing schedule
    anneal = LinearAnnealing(100)
    anneal['T'] = [(0.0, 1.5), (0.1, 1.5), (0.5, 1.)]
    anneal['Ncut_factor'] = [(0, 0.0), (.4, 1.)]
    # anneal['W_noise'] = [(0, 3.0), (.6, 0.0)]
    anneal['anneal_prior'] = False

    # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    model_params['W'] = np.zeros_like(model_params['W'])
    em.lparams = model_params
    import ipdb; ipdb.set_trace()  # breakpoint 646204a8 //
    em.run()

    dlog.close(True)
    pprint("Done")
