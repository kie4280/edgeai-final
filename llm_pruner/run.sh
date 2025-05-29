if [ "$1" == "prune" ]; then
    python llama3.py \
        --pruning_ratio 0.25 \
        --device cuda --eval_device cuda \
        --base_model meta-llama/Llama-3.2-3B-Instruct \
        --block_wise \
        --block_mlp_layer_start 4 --block_mlp_layer_end 28 \
        --block_attention_layer_start 4 --block_attention_layer_end 28 \
        --save_ckpt_log_name llama3_prune \
        --pruner_type taylor --taylor param_first \
        --max_seq_len 2048 \
        --save_model 

    echo "[FINISH] - Finish Pruning Model"

elif [ "$1" == "post_train" ]; then 
    python post_training.py --prune_model prune_log/llama3_prune/pytorch_model.bin \
        --data_path yahma/alpaca-cleaned \
        --lora_r 8 \
        --num_epochs 2 --learning_rate 1e-4 --batch_size 64 \
        --output_dir tune_log/llama3.2-intruct \
        --wandb_project llama_tune \
        --
    echo "[FINISH] - Finish Finetuning Model" 

elif [ "$1" == "result" ]; then 
    python result.py
fi
