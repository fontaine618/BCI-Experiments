#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=bci_sim
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=0:01:00
#SBATCH --array=0-119
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
#python -O mllk.py $SLURM_ARRAY_TASK_ID
#python -O ic_x.py $SLURM_ARRAY_TASK_ID
#python -O ic_y.py $SLURM_ARRAY_TASK_ID
python -O test.py $SLURM_ARRAY_TASK_ID

