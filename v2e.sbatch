#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH -t48:00:00
#SBATCH --output=session_v2e_%j.out


# Run.
PYTHONPATH=$PYTHONPATH:. python train_run.py -hs 512 --iterations 150000 --model_type rnn --ml 35

#optional arguments:
#  -h, --help            show this help message and exit
#  -hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
#                        hidden size dimmension
#  -it ITERATIONS, --iterations ITERATIONS
#  -pr PRINT_ITER, --print_iter PRINT_ITER
#  -pl PLOT_ITER, --plot_iter PLOT_ITER
#  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
#  -ml MAX_LENGTH, --max_length MAX_LENGTH
#  -mv MAX_VOCAB, --max_vocab MAX_VOCAB
#  -tf TEACHER_FORCE, --teacher_force TEACHER_FORCE
#  -sl SOURCE_FILE, --source_file SOURCE_FILE
#  -tl TARGET_FILE, --target_file TARGET_FILE
#  -mt MODEL_TYPE, --model_type MODEL_TYPE
#                        rnn, attn, ...
#  -md MODEL_DIRECTORY, --model_directory MODEL_DIRECTORY
#                       where to put our models

