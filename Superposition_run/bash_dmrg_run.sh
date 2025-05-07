#!/bin/bash

#SBATCH --partition=defq
#SBATCH --array=0-5%3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH --time=0-10:60:00
#SBATCH --job-name=DMRG_34_44
#SBATCH --output=Superposition_run/output/DMRG_Run_%x_%a.out
## #SBATCH --exclude=comp044

	
PYTHON=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/.env/bin/python3

SCRIPT=/gpfs01/home/ppzaj/python_projects/HF_Fermionic_State_Prepration/Superposition_run/dmrg_tenpy_V0.py


echo " BASH DMRG START "
$PYTHON -u $SCRIPT $SLURM_JOB_ID $SLURM_ARRAY_TASK_ID
echo " BASH DMRG END  "
