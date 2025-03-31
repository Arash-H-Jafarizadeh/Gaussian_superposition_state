#!/bin/bash

#SBATCH --partition=defq
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=36g
#SBATCH --time=0-23:33:33
#SBATCH --output=Superposition_run/output/Aash_Sophis_Parallel_Run_%a.out
## #SBATCH --output=Superposition_run/output/Bash_Superpos_Parallel_Run_%j_%a.out
## #SBATCH --exclude=comp044

	
PYTHON=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/.env/bin/python3

## SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition/bash_superpos_parallel_run.py
## SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition/sophis_superpos_run.py
SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition_run/sophis_superpos_run_V1.py


echo " BASH PARALLELED ARRAY START "
$PYTHON -u $SCRIPT $SLURM_ARRAY_TASK_ID $SLURM_JOB_ID
## $PYTHON $SCRIPT $SLURM_JOB_ID
echo " BASH PARALLELED ARRAY END  "
