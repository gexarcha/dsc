#!/bin/zsh
#SBATCH --job-name="DSCTVEMGIBBS"
#SBATCH -n 50
##SBATCH -N 9
##SBATCH --cpus-per-task=1
##SBATCH --ntasks-per-node=11
##SBATCH --ntasks-per-core=1
##SBATCH --mem-per-cpu=10G
##SBATCH
##SBATCH --nodelist="gold01,gold02,gold03,gold04,gold05,gold06,gold07,gold08,gold09" 
#SBATCH --time="15-20" 
##SBATCH --mincpus=11
#
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
#
# send mail to this address
#SBATCH --mail-user=georgios.exarchakis@uni-oldenburg.de
#SBATCH  --output="log_dsctvem%j.out"
#SBATCH  --error="log_dsctvem%j.err"
# /opt/anaconda/bin/mpiexec -iface bond0 python dsc_run.py
/opt/anaconda/bin/mpiexec -iface bond0 python dsc_run_audio.py
##/opt/anaconda/bin/mpiexec python dsc_run.py

