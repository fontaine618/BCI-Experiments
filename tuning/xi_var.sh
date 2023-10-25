#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=xi_var
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=8:00:00
#SBATCH --array=0-7
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
python -O xi_var.py $SLURM_ARRAY_TASK_ID
python -O xi_var_predict.py $SLURM_ARRAY_TASK_ID
