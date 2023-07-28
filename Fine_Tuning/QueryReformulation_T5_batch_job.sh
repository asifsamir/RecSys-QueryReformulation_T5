#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:p100:2
#SBATCH --time=9:0:0    
#SBATCH --mail-user=asifsamirgen@gmail.com
#SBATCH --mail-type=ALL


cd ~/$project/Projects/QueryReformulation_T5/Fine_Tuning/

module purge
module load gcc arrow/9 python/3.10

source ~/ENV/bin/activate


pip install transformers
pip install evaluate
pip install datasets
pip install accelerate -U
pip install tensorflow

python Finetune_T5_keyword.py