accelerate launch train_lora_dreambooth.py ^
  --pretrained_model_name_or_path="lambdalabs/sd-pokemon-diffusers"  ^
  --instance_data_dir="./data/Apple" ^
  --output_dir="./output/Apple3" ^
  --logging_dir="./logs/Apple" ^
  --instance_prompt="emoji" ^
  --train_text_encoder ^
  --resolution=256 ^
  --train_batch_size=8 ^
  --gradient_accumulation_steps=1 ^
  --learning_rate=1e-4 ^
  --learning_rate_text=5e-5 ^
  --color_jitter ^
  --lr_scheduler="constant" ^
  --lr_warmup_steps=0 ^
  --max_train_steps=1000 ^