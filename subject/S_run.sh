#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=bci_subject_S
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=10:00:00
#SBATCH --array=117, 121, 146, 151, 158, 171, 172, 177, 183
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
# Potential subjects: 111,114,115,122,143,145,154,155,159,160,166,178,179,183
# Subjects in Ma (2022): 114, 117, 121, 146, 151, 158, 171, 172, 177, 183
