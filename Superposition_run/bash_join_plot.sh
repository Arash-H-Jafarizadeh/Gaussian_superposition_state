#!/bin/bash

## #SBATCH --partition=devq ## #SBATCH --qos=dev
#SBATCH --partition=defq
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=30g
#SBATCH --time=0-21:11:22
#SBATCH --output=Superposition_run/output/Amps_JoinPlot_Run_%a.out
## #SBATCH --output=Superposition_run/output/X_JoinPlot_Run.out ##### for test
## #SBATCH --exclude=comp044

	
PYTHON=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/.env/bin/python3

SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition_run/join_data_run_V0.py
## SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition_run/plot_data_run_V1.py


echo " BASH JOIN & PLOT START "
$PYTHON -u $SCRIPT $SLURM_JOB_ID $SLURM_ARRAY_TASK_ID
echo " BASH JOIN & PLOT END  "
