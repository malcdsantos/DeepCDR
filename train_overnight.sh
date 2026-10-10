#!/bin/bash
#SBATCH --job-name=deepcdr-overnight
#SBATCH --partition=dedicated_big_bigmem
#SBATCH --account=lp_big_wice_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/data/leuven/395/vsc39540/DeepCDR_working/logs/overnight-%j.out

#SBATCH --error=/data/leuven/395/vsc39540/DeepCDR_working/logs/overnight-%j.err

set -e

source /etc/profile
module load cluster/wice/dedicated_big_bigmem

module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate /data/leuven/395/vsc39540/conda_envs/deepcdr-original

cd /data/leuven/395/vsc39540/DeepCDR_working/prog

export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1

python run_DeepCDR_classify.py \
  --experiment-dir ../checkpoint/experiments/resume-test01 \
  --resume
