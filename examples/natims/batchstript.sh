#!/bin/zsh
#SBATCH --job-name="DSCTVEMGIBBS"
#SBATCH -n 50
#SBATCH --time="15-20" 
#
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
#
# send mail to this address
#SBATCH --mail-user=georgios.exarchakis@uni-oldenburg.de
#SBATCH  --output="log_dsctvem%j.out"
#SBATCH  --error="log_dsctvem%j.err"
#mpiexec -iface bond0 python dsc_run.py

