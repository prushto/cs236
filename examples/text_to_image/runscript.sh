export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="logo-output-gpt_gen_desc_logos"
export HUB_MODEL_ID="prushton/logo-lora-real-world"
export DATASET_NAME="mdass/gpt_gen_desc_logos"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --caption_column="description" \
  --max_train_samples=1 \
  --validation_epochs=100 \
  --validation_prompt="University logo featuring a tree in red and white colors" \
  --seed=1337
