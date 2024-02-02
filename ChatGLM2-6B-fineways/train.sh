CUDA_VISIBLE_DEVICES=0 /home/best/miniconda3/envs/llm/bin/python main.py \
    --do_train \
    --train_file train.json \
    --validation_file dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path "/home/best/ChatGLM2-6B/chatglm2-6b/" \
    --output_dir "./lora/" \
    --finetuning_type lora \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate 5e-4
        
CUDA_VISIBLE_DEVICES=0 /home/best/miniconda3/envs/llm/bin/python main.py \
    --do_eval \
    --train_file train.json \
    --validation_file dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path "/home/best/ChatGLM2-6B/chatglm2-6b/" \
    --checkpoint "./lora/checkpoint-10/" \
    --output_dir "./lora/" \
    --finetuning_type lora \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-4

CUDA_VISIBLE_DEVICES=0 /home/best/miniconda3/envs/llm/bin/python main.py \
    --do_train \
    --train_file train.json \
    --validation_file dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path "/home/best/ChatGLM2-6B/chatglm2-6b/" \
    --output_dir "./p_tuning/" \
    --finetuning_type p_tuning \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate 2e-2 \
    --pre_seq_len 128

CUDA_VISIBLE_DEVICES=0 /home/best/miniconda3/envs/llm/bin/python main.py \
    --do_eval \
    --train_file train.json \
    --validation_file dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path "/home/best/ChatGLM2-6B/chatglm2-6b/" \
    --checkpoint "./p_tuning/checkpoint-3000/" \
    --output_dir "./p_tuning/" \
    --finetuning_type p_tuning \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 2e-2 \
    --pre_seq_len 128