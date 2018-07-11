
# Introduction

This package contains all the source code to reproduce the numerical
experiments described in the paper Discrete Sparse Coding. 

## Software dependencies 
 
 * Python (>= 2.6)
 * NumPy (reasonably recent)
 * SciPy (reasonably recent)
 * pytables (reasonably recent)
 * mpi4py (>= 1.3)

## Overview 

pulp/       - Python library/framework for MPI parallelized 
              EM-based algorithms. The models' implementations
              can be found in pulp/em/camodels/.

examples/   - Small examples for initializing and running the models



## Running

To run the barstest experiment:

  $ cd examples/barstest
  $ python dsc_run.py

To run the natural images experiment:
  $ cd ../natims
  $ python dsc_run.py

To run the spikes experiment:
  $ cd ../spikes
  $ python dsc_on_hc1_run.py

To run the audio experiment:
  $ cd ../audio
  $ python dsc_run_audio.py

Some of this experiments are too big to run in a single workstation
and should be executed on a cluster. Running our experiments on the 
cluster largely depends on the configuration. Example batch files 
for our cluster (slurm based) configuration (GOLD cluster - Uni Oldenburg) are
given in examples/<experiment name>/batchscript.sh

## Results/Output

The results produced by the code are stored in a 'results.h5' file 
under "./output/.../". The file stores the model parameters (e.g., W, pi etc.) 
for each EM iteration performed. To read the results file, you can use
openFile function of the standard tables package in python. Moreover, the
results files can also be easily read by other packages such as Matlab etc.

## Running on a parallel architecture 

The code uses MPI based parallelization. If you have parallel resources
(i.e., a multi-core system or a compute cluster), the provided code can make a 
use of parallel compute resources by evenly distributing the training data 
among multiple cores.

To run the same script as above, e.g., 

a) On a multi-core machine with 32 cores:

 $ mpirun -np 32 python bars-run-all.py param-bars-<...>.py

b) On a cluster:

 $ mpirun --hostfile machines python bars-run-all.py param-bars-<...>.py

 where 'machines' contains a list of suitable machines.

See your MPI documentation for the details on how to start MPI parallelized 
programs.
