if [ "$1" == "finetune-3B" ]; then
    tune run lora_finetune_single_device \
        --config llama3_2_3B_finetune.yaml

elif [ "$1" == "download-1B" ]; then
    tune download meta-llama/Llama-3.2-1B-Instruct \
        --output-dir ckpt/Llama-3.2-1B-Instruct \
        --hf-token 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

elif [ "$1" == "download-3B" ]; then
    tune download meta-llama/Llama-3.2-3B-Instruct \
        --output-dir ckpt/Llama-3.2-3B-Instruct \
        --hf-token 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

elif [ "$1" == "distill" ]; then
    tune run knowledge_distillation_single_device \
        --config llama3.2-Instruct-distill.yaml 

elif [ "$1" == "result" ]; then 
    python result.py \
        --tuned_dir tune_log/llama4.2-1B/ \
        --lora_model torchtune/llama3.2_3B_to_1B/epoch_4
fi

# if [ "$1" == "finetune" ]; then 
#     python post_training.py \
#         --model "meta-llama/Llama-3.2-1B" \
#         --lora_r 8 \
#         --num_epochs 2 --learning_rate 1e-4 --batch_size 64 \
#         --output_dir tune_log/llama3.2-1B\
#         --wandb_project distill-v2
#     echo "[FINISH] - Finish Finetuning Model" 

# elif [ "$1" == "distill" ]; then 
#     python distill.py \
#         --model "meta-llama/Llama-3.2-1B" \
#         --lora_r 8 \
#         --num_epochs 2 --learning_rate 1e-4 --batch_size 64 \
#         --output_dir tune_log/llama3.2-1B\
#         --wandb_project distill-v2
#     echo "[FINISH] - Finish Finetuning Model" 