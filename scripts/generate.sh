if [ "$1" = "-m" ] && [ -n "$2" ]; then
  MODEL_NAME_OR_PATH=$2
else
  echo "Usage: $0 -m <model_path>"
  exit 1
fi

# Basic setting
export CUDA_LAUNCH_BLOCKING=1

# Training arguments
# ========================================================================
OUTPUT_DIR="output/"
DATASET_DIR="dataset/ConvAI2/llama3/"
K_HOP=3
N_TURNS=5
TAU=0.4
TOP_K_RELATIONS=5
# ========================================================================


python3 src/generate.py \
  --data_dir $DATASET_DIR \
  --processed_data_dir $DATASET_DIR/processed_valid_3-hop_5-turns/ \
  --data_name valid_self_original_coherence.pkl \
  --processed_data_name processed_valid_self_original.pt \
  --output_dir $OUTPUT_DIR \
  --model_name_or_path $MODEL_NAME_OR_PATH/model \
  --tokenizer_name_or_path $MODEL_NAME_OR_PATH/tokenizer \
  --process_mode single_filter \
  --k_hop $K_HOP \
  --directed \
  --no-reverse_edge \
  --batch_size 1 \
  --tau $TAU \
  --top_k_relations $TOP_K_RELATIONS
