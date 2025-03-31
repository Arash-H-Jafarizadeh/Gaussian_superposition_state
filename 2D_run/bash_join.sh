#!/bin/bash

#SBATCH --partition=defq
## #SBATCH --array=1 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3g
#SBATCH --time=0-00:01:00
#SBATCH --output=data/Join_Data__%j.out

	
PYTHON=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/.env/bin/python3

SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/bash_join_run.py
### SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/bash_plot_run.py


echo "BASH JOIN started"
$PYTHON $SCRIPT $SLURM_JOB_ID
echo "BASH JOIN ended"
