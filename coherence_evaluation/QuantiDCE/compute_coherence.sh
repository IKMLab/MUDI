SEED=71
# EVAL_MODE=separate

cd ..


INPUT_FILE_PATH=""
OUTPUT_FILE_PATH=""


# Computes human correlations
python quantidce_inference.py \
    --model bert_metric \
    --checkpoint_dir_path ./output/${SEED}/bert_metric_kd_finetune \
    --checkpoint_file_name model_best_kd_finetune_loss.ckpt \
    --pretrained_model_name bert-base-uncased \
    --input_file_path $INPUT_FILE_PATH \
    --output_file_path $OUTPUT_FILE_PATH \
