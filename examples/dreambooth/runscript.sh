export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="myra_db"
export OUTPUT_DIR="myra_output_3000"
export HUB_MODEL_ID="prushton/dreambooth-myra-3000"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of myra" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --gradient_checkpointing \
  --checkpointing_steps=1000 \
  --use_8bit_adam
