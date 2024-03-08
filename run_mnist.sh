#!/bin/bash

#SBATCH --job-name=pacbayes
#SBATCH --mem-per-cpu=64000M
#SBATCH --time=0-5:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pbb
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

kl_penaltys=(2.0 1.0 0.667 0.5 0.4 0.333 0.286) 

for kl_penalty in ${kl_penaltys[*]}; do

    python run_exp.py --name_data="mnist" \
                    --objective="fquad" \
                    --prior_type="learnt" \
                    --model="fcn" \
                    --kl_penalty=$kl_penalty &

    python run_exp.py --name_data="mnist" \
                    --objective="flamb" \
                    --prior_type="learnt" \
                    --model="fcn" \
                    --kl_penalty=$kl_penalty &

    python run_exp.py --name_data="mnist" \
                    --objective="fclassic" \
                    --prior_type="learnt" \
                    --model="fcn" \
                    --kl_penalty=$kl_penalty &

    python run_exp.py --name_data="mnist" \
                    --objective="bbb" \
                    --prior_type="learnt" \
                    --model="fcn" \
                    --kl_penalty=$kl_penalty &

    wait
done