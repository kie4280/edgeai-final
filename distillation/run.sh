if [ "$1" == "prune-block" ]; then
    python llama3.py \
        --pruning_ratio 0.2 \
        --device cuda --eval_device cuda \
        --base_model meta-llama/Llama-3.2-3B-Instruct \
        --block_wise \
        --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
        --block_attention_layer_start 4 --block_attention_layer_end 28 \
        --save_ckpt_log_name llama3.2_prune_1 \
        --pruner_type taylor --taylor param_first \
        --max_seq_len 2048 \
        --save_model 

    echo "[FINISH] - Finish Pruning Model"

elif [ "$1" == "prune-layer" ]; then
    python llama3.py \
        --pruning_ratio 0.01 \
        --device cuda --eval_device cuda \
        --base_model meta-llama/Llama-3.2-3B-Instruct \
        --layer_wise \
        --save_ckpt_log_name llama3_prune \
        --pruner_type taylor --taylor param_first \
        --max_seq_len 2048 \
        --save_model 

    echo "[FINISH] - Finish Pruning Model"

elif [ "$1" == "finetune" ]; then 
    python post_training.py \
        --model "meta-llama/Llama-3.2-1B" \
        --lora_r 8 \
        --num_epochs 2 --learning_rate 1e-4 --batch_size 64 \
        --output_dir tune_log/llama3.2-1B\
        --wandb_project distill 
    echo "[FINISH] - Finish Finetuning Model" 


elif [ "$1" == "prune_train" ]; then
    for ((i=1; i<=TOTAL_STEPS; i++)); do
        echo "========== [STEP $i] PRUNING =========="
        python llama3.py \
            --pruning_ratio 0.1 \
            --device cuda --eval_device cuda \
            --base_model meta-llama/Llama-3.2-3B-Instruct \
            --pruned True \
            --tuned_dir tune_log/llama3_tuned_$(i-1) \
            --block_wise \
            --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
            --block_attention_layer_start 4 --block_attention_layer_end 28 \
            --save_ckpt_log_name llama3_prune \
            --pruner_type taylor --taylor param_first \
            --max_seq_len 2048 \
            --save_model 

        CKPT="$PRUNE_PATH/pytorch_model.bin"

        echo "========== [STEP $i] FINETUNING =========="
        TUNE_OUTPUT="tune_log/${SAVE_PREFIX}_step${i}_tuned"
        python post_training.py \
            --prune_model prune_log/llama3_prune/pytorch_model.bin  \
            --data_path yahma/alpaca-cleaned \
            --lora_r 8 \
            --num_epochs 2 --learning_rate 1e-4 --batch_size 64 \
            --output_dir tune_log/llama3_tuned_$i\
            --wandb_project ${SAVE_PREFIX}_tune

        CURRENT_MODEL="${TUNE_OUTPUT}/pytorch_model.bin"
    done

    echo "[FINISH] - Iterative Pruning + Finetuning Complete"

elif [ "$1" == "result" ]; then 
    python result.py \
        --tuned_dir tune_log/llama3.2-1B/
fi
