#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=bci_subject_S
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=10:00:00
#SBATCH --array=114,117,121,146,151,158,171,172,177,183
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
python -O S_train.py $SLURM_ARRAY_TASK_ID
python -O S_mllk.py $SLURM_ARRAY_TASK_ID
python -O S_importance.py $SLURM_ARRAY_TASK_ID
# Subjects in Ma (2022): 114,117,121,146,151,158,171,172,177,183
# From Thompson (2014):
# - Controls: 114,117,121,171,172,177,183
# - ALS: 146,151,158,
