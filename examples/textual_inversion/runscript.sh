export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data_new"
export HUB_MODEL_ID="prushton/text-inv-myra_fridec8"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<myra>" \
  --initializer_token="myra" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_myra" \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --validation_prompt="A picture of <myra>" \
  --num_validation_images=4 \
  --validation_steps=1000
