#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=bci_sim_importance
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=5:00:00
#SBATCH --array=0-5
#SBATCH --account=stats_dept1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=16g
#SBATCH --output=/home/%u/logs/%x-%j.log
# The application(s) to execute along with its input arguments and options:
module load python/3.10.4
source /home/simfont/venvs/bci/bin/activate
#python -O train.py $SLURM_ARRAY_TASK_ID
#python -O importance.py $SLURM_ARRAY_TASK_ID
#python -O posterior.py $SLURM_ARRAY_TASK_ID
#python -O drop_one.py $SLURM_ARRAY_TASK_ID
python -O just_one.py $SLURM_ARRAY_TASK_ID

