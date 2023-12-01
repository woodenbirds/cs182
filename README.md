# Fine-Tuning Lambda Diffusion Model for Emojis Generation from Text

### You  can use google colab to train the whole model.

## Overview

The main goal of this project is to generating customized emojis in different software style using a fine-tuned stable diffusion model. The methodology includes data preprocessing from the Full Emoji Image Dataset and details the integration of LoRA with the base model.

## Fine-tuning Instruction

We provide the corresponding code and dataset in [Final](Final)  folder. You can run files in [lora_finetune_attention_weight.ipynb](Final/lora_finetune_attention_weight.ipynb) to reproduce the model.

### Requirement

For the training requirement, we recommend using colab to run this file. You can upload the whole [Final](Final) folder to colab and run [lora_finetune_attention_weight.ipynb](Final/lora_finetune_attention_weight.ipynb). In order to save data and visualize it, we recommend that you use wanda in the code to synchronize data. For this, you may need to register a relevant account. 

### Training

During the training, you just need to follow the instruction in the [lora_finetune_attention_weight.ipynb](Final/lora_finetune_attention_weight.ipynb).

## Dataset

We used [Full Emoji Image Dataset] (https://www.kaggle.com/datasets/subinium/emojiimage-dataset) on kaggle as our training and validation dataset. We use [test.ipynb](test.ipynb) to preprocess the data into the form we need, which are stored in the "Final" folder. You can also try to process them by yourself.

