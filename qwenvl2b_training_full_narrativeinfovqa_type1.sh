# Training script for QwenVL with NarrativeInfoVQA dataset
# 2 * 23GiB; 2.3s/it
PYTORCH_ALLOC_CONF='expandable_segments:True' \
CUDA_LAUNCH_BLOCKING=1 \
NPROC_PER_NODE=2 \
MAX_PIXELS=1003520 \
MASTER_PORT=29502 \
CUDA_VISIBLE_DEVICES=2,3 \
megatron sft \
--model /media/workspace/thangdd_workspace/llm_checkpoints/Qwen_Qwen3-VL-2B-Instruct \
--dataset '/media/workspace/thangdd_workspace/InfographicDataPaper/NarrativeInfoVQA/ms_swift_format_train/narrativeinfovqa_train_n20000.jsonl' \
--val_dataset '/media/workspace/thangdd_workspace/InfographicDataPaper/msswift_infographicvqa_dataset/infographicvqa_val.jsonl' \
--bf16 true \
--train_type full \
--tensor_model_parallel_size 1 \
--sequence_parallel false \
--freeze_llm false \
--freeze_vit true \
--freeze_aligner true \
--packing true \
--micro_batch_size 1 \
--global_batch_size 10 \
--recompute_granularity selective \
--finetune true \
--cross_entropy_loss_fusion true \
--lr 3e-5 \
--lr_warmup_fraction 0.05 \
--train_iters 2000 \
--save megatron_output/Qwen3-VL-2B-Instruct-NarrativeInfoVQA \
--save_interval 200 \
--vit_gradient_checkpointing true \
--max_length 2048 \
--num_workers 4 \
--no_save_optim true \
--dataset_num_proc 16 \
--no_save_rng true \
--report_to wandb \
--wandb_project narrativeinfovqa \
--wandb_exp_name Qwen3-VL-2B-Instruct-NarrativeInfoVQA_2kiters_fullfinetune_falsellm_truevit_truealigner

# --max_epochs 1 \
# --train_iters 2000 \
# --val_dataset '/media/workspace/thangdd_workspace/InfographicDataPaper/NarrativeInfoVQA/ms_swift_format_test/narrativeinfovqa_val_no_unans.jsonl' \