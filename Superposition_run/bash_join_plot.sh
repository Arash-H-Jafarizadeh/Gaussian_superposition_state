#!/bin/bash

#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8g
#SBATCH --time=0-00:21:22
#SBATCH --output=Superposition_run/output/Z_JoinPlot_Run.out 


## #SBATCH --partition=devq ## #SBATCH --qos=dev
## #SBATCH --array=0-9
## #SBATCH --output=Superposition_run/output/Amps_JoinPlot_Run_%a.out
## #SBATCH --output=Superposition_run/output/X_JoinPlot_Run.out 
## #SBATCH --exclude=comp007
## #SBATCH --nodelist=comp007

	
PYTHON=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/.env/bin/python3

SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition_run/join_data_run_V0.py

## SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition_run/plot_data_run_V2.py


echo " BASH JOIN & PLOT START "
$PYTHON -u $SCRIPT $SLURM_JOB_ID $SLURM_ARRAY_TASK_ID
echo " BASH JOIN & PLOT END  "
