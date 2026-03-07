#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=bci_subject_V
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=simfont@umich.edu
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --account=statsresearch_cr_default
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/%u/logs/BCI/%x-%j.log
# The application(s) to execute along with its input arguments and options:
module load python/3.11.2
source /storage/work/spf5519/BCI/bci/bin/activate
python -O V_train.py 117 $SLURM_ARRAY_TASK_ID
python -O V_mllk.py 117 $SLURM_ARRAY_TASK_ID
python -O V_ics.py 117 $SLURM_ARRAY_TASK_ID

