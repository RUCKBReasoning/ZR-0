set -e

# while [[ $# -gt 0 ]]; do
#     key="$1"
#     case $key in
#         --num_gpu) NUM_GPU="$2"; shift ;;
#         --base_model_path) BASE_MODEL_PATH="$2"; shift ;;
#         --fast_tokenizer_path) FAST_TOKENIZER_PATH="$2"; shift ;;
#         --per_device_bs) PER_DEVICE_BS="$2"; shift ;;
#         --epochs) EPOCHS="$2"; shift ;;
#         --save_ckpt_interval) SAVE_CKPT_INTERVAL="$2"; shift ;;
#         --peak_lr) PEAK_LR="$2"; shift ;;
#         --min_lr_rate) MIN_LR_RATE="$2"; shift ;;
#         --tensorboard_log_dir) TENSORBOARD_LOG_DIR="$2"; shift ;;
#         --output_ckpt_dir) OUTPUT_CKPT_DIR="$2"; shift ;;
#         --dataset_entries) DATASET_ENTRIES="$2"; shift ;;
#         *) echo "Unknown option $1"; exit 1 ;;
#     esac
#     shift
# done


set -e
while [ "$#" -gt 0 ]; do
    case "$1" in
        --num_gpu) NUM_GPU="$2"; shift 2 ;;
        --base_model_path) BASE_MODEL_PATH="$2"; shift 2 ;;
        --fast_tokenizer_path) FAST_TOKENIZER_PATH="$2"; shift 2 ;;
        --per_device_bs) PER_DEVICE_BS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --save_ckpt_interval) SAVE_CKPT_INTERVAL="$2"; shift 2 ;;
        --peak_lr) PEAK_LR="$2"; shift 2 ;;
        --min_lr_rate) MIN_LR_RATE="$2"; shift 2 ;;
        --tensorboard_log_dir) TENSORBOARD_LOG_DIR="$2"; shift 2 ;;
        --output_ckpt_dir) OUTPUT_CKPT_DIR="$2"; shift 2 ;;
        --dataset_entries) DATASET_ENTRIES="$2"; shift 2 ;;
        *) echo "Unknown option $1"; exit 1 ;;
    esac
done

accelerate launch \
    --num_processes $NUM_GPU \
    --config_file ./accelerate_configs/accelerate_config.yaml \
    train_vla.py \
    --vlm_name_or_path $BASE_MODEL_PATH \
    --action_expert_name_or_path $BASE_MODEL_PATH \
    --FAST_tokenizer_path $FAST_TOKENIZER_PATH \
    --per_device_train_batch_size $PER_DEVICE_BS \
    --seed 42 \
    --epochs $EPOCHS \
    --save_ckpt_interval $SAVE_CKPT_INTERVAL \
    --save_step_interval 1000000 \
    --peak_learning_rate $PEAK_LR \
    --min_lr_rate $MIN_LR_RATE \
    --tensorboard_log_dir $TENSORBOARD_LOG_DIR \
    --output_ckpt_dir $OUTPUT_CKPT_DIR \
    --tune_vlm \
    --tune_action_expert \
    --loss_type "vlm_and_action" \
    --vlm_loss_weight 1.0 \
    --action_expert_loss_weight 5.0 \
    --lr_scheduler "cosine" \
    --dataset_entries $DATASET_ENTRIES \
    --window_size 1 \
    --action_horizon 32 \
    --max_pad_state_and_action_length 64
