export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
export CONVERTED_MODEL_PATH="converted_models/llama-2-7b"

if [[ "$1" == "--skip-convert" ]]; then
    SKIP=true
fi

if [ "$SKIP" = false ]; then
    export COAT_PATH=$(pip show coat | grep "Editable project location" | awk -F': ' '{print $2}')
    echo "COAT package is located at: $COAT_PATH"

    python $COAT_PATH/coat/activation/models/coat_llama_convert_from_hf.py \
        --model_name $MODEL_NAME \
        --save_path $CONVERTED_MODEL_PATH \
        --quantize_model true \
        --fabit E4M3 \
        --fwbit E4M3 \
        --fobit E4M3 \
        --bwbit E5M2 \
        --babit E5M2 \
        --bobit E5M2 \
        --group_size 16
fi
# We double the batch size here
torchrun --nproc_per_node=8 --master_port=20001 toolbench/train/train_fp8.py \
    --model_name_or_path $MODEL_NAME  \
    --fp8_model_name_or_path $CONVERTED_MODEL_PATH  \
    --data_path  data/toolllama_G123_dfs_train.json \
    --eval_data_path  data/toolllama_G123_dfs_eval.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir toolllama \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --prediction_loss_only \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'CoatLlamaDecoderLayer' \
    --tf32 True \
    --source_model_max_length 4096 \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --report_to wandb \
    --quantize_model true \
    --fabit E4M3 \
    --fwbit E4M3 \
    --fobit E4M3 \
    --bwbit E5M2 \
    --babit E5M2 \
    --bobit E5M2 \
    --group_size 16 \
    --first_order_expansion true \
    --second_order_expansion true \
    --first_order_bit E4M3 \
    --second_order_bit E4M3 \
    --qgroup_size 128 \
    --expand_min 16
