# get current directory
DIR=$(pwd)

# run the python script
cd py/launchers
# set the jax memory allocator
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
python learn.py \
    --target cifar10 \
    --box_anneal 0 \
    --diagonal_anneal 0 \
    --bs 256 \
    --plot_bs 64 \
    --visual_freq 1000 \
    --save_freq 5000 \
    --shuffle_dataset 1 \
    --overfit 0 \
    --conditional 0 \
    --n_epochs 150000 \
    --distill_steps 120000 \
    --distill_delta 0.01 \
    --n 50000 \
    --d 32 \
    --n_neurons 256 \
    --n_hidden 4 \
    --act "swish" \
    --learning_rate 0.0001 \
    --decay_steps 0 \
    --warmup_steps 0 \
    --device_type cuda \
    --wandb_name "cifar10_flow" \
    --wandb_entity "jc-bao" \
    --wandb_project FlowMapMatching \
    --output_name cifar10_run \
    --output_folder "../../results" \
    --base gaussian \
    --loss_type lagrangian \
    --anneal_steps 0 \
    --gaussian_scale 1 \
    --tmax 1 \
    --tmin 0 \
    --class_dropout 0 \
    --slurm_id 0

# python learn.py --target mnist --box_anneal 0 --diagonal_anneal 0 --bs 256 --n_epochs 250000 --distill_steps 250000 --n 1281167 --d 32 --n_neurons 256 --learning_rate 0.0001 --decay_steps 1000 --warmup_steps 1000 --device_type cuda --wandb_project elliptic_learning --output_name mnist_run

# go back to the original directory
cd $DIR
