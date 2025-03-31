#!/bin/bash

#SBATCH --partition=defq
## #SBATCH --array=0-19 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8g
#SBATCH --time=0-00:30:00
#SBATCH --output=data/Bash_Superposition_Run_%j.out
## #SBATCH --exclude=comp044

	
PYTHON=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/.env/bin/python3

SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/bash_superposition_1D_run.py


echo " BASH PARALLEL start "
## $PYTHON $SCRIPT $SLURM_ARRAY_TASK_ID $SLURM_JOB_ID
$PYTHON $SCRIPT $SLURM_JOB_ID
echo " BASH PARALLEL end  "
