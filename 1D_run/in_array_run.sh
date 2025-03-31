#!/bin/bash

#SBATCH --partition=defq
#SBATCH --array=1-3 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=20g
#SBATCH --time=4-00:00:00
#SBATCH --output=data/Fermionic_RUN_%a_%j.out

	
PYTHON=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/.env/bin/python3
### SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/python_run.py
SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/in_array_run.py


echo " BASH start "
$PYTHON $SCRIPT $SLURM_ARRAY_TASK_ID $SLURM_JOB_ID
echo " BASH  end  "

