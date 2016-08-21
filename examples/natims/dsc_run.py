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
from pulp.utils.autotable import AutoTable
import tables
from pulp.utils.parallel import pprint, stride_data

from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
# from pulp.visualize.gui import GUI, RFViewer, YTPlotter

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing
from pulp.utils.vc import VClog

output_path = create_output_path()
run_tag = "run_"+output_path[output_path.find('.py.')+4:-1].replace(":","_")
gitlog=VClog()
if gitlog.repo.is_dirty():
    gitlog.autocommit(MPI.COMM_WORLD,run_tag=run_tag)
else:
    if MPI.COMM_WORLD==0:
        gitlog.tag(run_tag)

def rank0only(func, *args, **kargs):
    if MPI.COMM_WORLD.rank == 0:
        func(*args, **kargs)
    MPI.COMM_WORLD.barrier()


# np.__config__.show()
# Number of datapoints to generate
N = 200000

# Each datapoint is of D = size*size
size = 16

# Diemnsionality of the model
D = size ** 2    # dimensionality of observed data
H = 300

# Approximation parameters for Expectation Truncation
Hprime = 8
gamma = 5

# Define Desired States
# states=np.array([-1.,0.,1.])
states=np.array([0.,1.])
# states=np.array([0.,1.,2.,3.,4.])

# Import and instantiate a model
from pulp.em.dmodels.dsc_et import DSC_ET
model = DSC_ET(D, H, Hprime, gamma,states=states, to_learn=['W','pi','sigma'])
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
    # data_fname = "/users/ml/xoex6879/data/VanHaterenImages/natims_conv_1700_p26.h5"
    data_fname = "/users/ml/xoex6879/data/VanHaterenImages/natims_conv_1700_p16_95var.h5"
    with tables.open_file(data_fname, 'r') as data_h5:
        N_file = data_h5.root.wdata.shape[0]
        if N_file < N:
            dlog.progress(
                "WARNING: N={} chosen but only {} data points available. ".format(N, N_file))
            N = N_file

        first_y, last_y = stride_data(N)
        my_y = data_h5.root.wdata[first_y:last_y]
        data_h5.close()

    my_data = {
        'y': my_y,
    }
    pprint("{}".format(my_y[8,:10]))
    # Configure DataLogger
    print_list = ('T', 'Q', 'pi', 'sigma', 'N','N_use', 'L','W_noise','sigma_noise','pi_noise','prior_mass')
    h5_list = ('W', 'pi', 'sigma', 'y', 'N',  'N_use','prior_mass','states','Hprime','H','gamma')
    # dlog.set_handler(['L'], YTPlotter)
    dlog.set_handler(print_list, TextPrinter)
    dlog.set_handler(print_list, StoreToTxt, output_path + '/terminal.txt')
    dlog.set_handler(h5_list, StoreToH5, output_path +'/result.h5')

    # dlog.append('y',my_data['y'])
    dlog.append('N',N)
    dlog.append('Hprime',Hprime)
    dlog.append('H',H)
    dlog.append('gamma',gamma)
    dlog.append('states',states)
    # Prepare model...
    model_params = model.standard_init(my_data)
#    init_sparseness = 1/float(H)
 #   model_params['pi'] = np.array([1- init_sparseness ,init_sparseness])
    dlog.append_all(model_params)

    # Choose annealing schedule
    anneal = LinearAnnealing(200)
    # anneal['T'] = [(0.0, 1.1),(0.1,1.1), (0.5, 1.)]
    anneal['T'] = [(0.0, 2.), (0.05,2.), (0.4, 1.)]
    # anneal['W_noise'] = [(0.0, 0.), (0.05,0.), (0.15,1.), (0.3,3.), (0.5, 0.)]
    # anneal['W_noise'] = [(0.0, 0.2), (0.05,0.2), (0.5, 0.)]
    # anneal['W_noise'] = [(0.0, 1.),(0.1, 1./10.0), (0.5, 0.0)]
    # anneal['T'] = [(0.0, 1.),(0.1,1.), (0.5, 1.)]
    anneal['Ncut_factor'] = [(0.0, 0.),(0.1, 0.0),(0.3, 1.0)]
    # anneal['Ncut_factor'] = [(0.0, 0.1),(0.5, 1.0)]
    # anneal['W_noise'] = [(0.0, 1./10),(0.1, 1./10.0), (0.5, 0.0)]
    anneal['anneal_prior'] = False

    # Create and start EM annealing
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    em.run()

    dlog.close(True)
    pprint("Done")
