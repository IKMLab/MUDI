# Basic setting
export CUDA_LAUNCH_BLOCKING=1

# Training arguments
# ========================================================================
CHECKPOINT_DIR="checkpoints/gnn/pretraining/RCC/"
DATASET_DIR="dataset/RCC/reddit_conversations_v1.0_5turns/"
GNN_LAYER_TYPE="GATv2"
K_HOP=3
N_LAYERS=2
N_HEADS=4
DIM=512
# ========================================================================

# If you want to use the Weights & Biases (Wandb) logging, please set the following variables:
# Please set the --wandb_entity $WANDB_ENTITY \ --wandb_project $WANDB_PROJECT \ --wandb_run_name $WANDB_RUN_NAME \ --wandb to enable the Wandb logging.
# ========================================================================
# Entity name for Weights & Biases (Wandb)
WANDB_ENTITY=""
# Project name for Weights & Biases (Wandb)
WANDB_PROJECT=""
# Run name for Weights & Biases (Wandb)
WANDB_RUN_NAME="${GNN_LAYER_TYPE}_${N_LAYERS}layers_${K_HOP}hop_${DIM}dim_${N_HEADS}heads_RCC-5turns"
# ========================================================================

python3 src/training/train_gnn.py \
    --data_dir $DATASET_DIR \
    --processed_train_data_dir $DATASET_DIR/processed_train/ \
    --processed_valid_data_dir $DATASET_DIR/processed_valid/ \
    --train_data_name train.pkl \
    --valid_data_name dev.pkl \
    --processed_train_data_name processed_train.pt \
    --processed_valid_data_name processed_valid.pt \
    --ckpt_dir $CHECKPOINT_DIR \
    --train_mode pretraining \
    --k_hop $K_HOP \
    --layer_type $GNN_LAYER_TYPE \
    --num_layers $N_LAYERS \
    --num_heads $N_HEADS \
    --embedding_dim $DIM \
    --batch_size 512 \
    --epochs 100 \
    --lr 0.00002 \
    --weight_decay 0.01 \
    --optimizer adamw \
    --endure_times 10 \
