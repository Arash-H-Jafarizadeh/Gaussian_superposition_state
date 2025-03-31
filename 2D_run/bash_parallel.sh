#!/bin/bash

#SBATCH --partition=defq
#SBATCH --array=0-19 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=35g
#SBATCH --time=3-00:00:00
#SBATCH --output=parallel_data_2D/Bash_Parallel_Run_%a_%j.out
#SBATCH --exclude=comp044

	
PYTHON=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/.env/bin/python3

SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/bash_parallel_2D_run.py


echo " BASH PARALLEL start "
$PYTHON $SCRIPT $SLURM_ARRAY_TASK_ID $SLURM_JOB_ID
echo " BASH PARALLEL end  "
