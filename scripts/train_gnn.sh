# Basic setting
export CUDA_LAUNCH_BLOCKING=1

# Training arguments
# ========================================================================
# Path to the pretrained graph encoder. Please set the state_dict.pt file path.
PRETRAINED_GRAPH_ENCODER_PATH=""
CHECKPOINT_DIR="checkpoints/gnn/finetuning/ConvAI2/"
DATASET_DIR="dataset/ConvAI2/llama3/"
GNN_LAYER_TYPE="DialogueGAT"
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
WANDB_RUN_NAME="${GNN_LAYER_TYPE}_${N_LAYERS}layers_${K_HOP}hop_${DIM}dim_${N_HEADS}heads_ConvAI2"
# ========================================================================

python3 src/training/train_gnn.py \
    --data_dir $DATASET_DIR \
    --processed_train_data_dir $DATASET_DIR/processed_train_$K_HOP-hop/ \
    --processed_valid_data_dir $DATASET_DIR/processed_valid_$K_HOP-hop/ \
    --train_data_name train_self_original_coherence.pkl \
    --valid_data_name valid_self_original_coherence.pkl \
    --processed_train_data_name processed_train_self_original_coherence.pt \
    --processed_valid_data_name processed_valid_self_original_coherence.pt \
    --process_mode single_filter \
    --ckpt_dir $CHECKPOINT_DIR \
    --k_hop $K_HOP \
    --layer_type $GNN_LAYER_TYPE \
    --num_layers $N_LAYERS \
    --num_heads $N_HEADS \
    --pretrained_utterance_encoder none \
    --pretrained_model_path $PRETRAINED_GRAPH_ENCODER_PATH \
    --directed \
    --no-reverse_edge \
    --embedding_dim $DIM \
    --batch_size 512 \
    --epochs 100 \
    --lr 0.00002 \
    --weight_decay 0.01 \
    --optimizer adamw \
    --endure_times 15 \
    --coh_rel_cls_weight 1.5 \
    --link_prediction_weight 1.2 \
    --next_resp_type_direct_weight 1.5 \
    --next_resp_type_seq_weight 1.5
