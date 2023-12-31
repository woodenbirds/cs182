# Fine-Tuning Stable Diffusion Model for Emojis Generation from Text

### You can use google colab to train the whole model.
### We have tried about 5 different ways, you can just try the fourth one and the fifth one since they perform better than the other three.

## Overview

The main goal of this project is to generating customized emojis in different software style using a fine-tuned stable diffusion model. The methodology includes data preprocessing from the Full Emoji Image Dataset and details the integration of LoRA with the base model.

<img src="https://github.com/woodenbirds/cs182/blob/master/imgs/shock_origin.png" width="10%"> <img src="https://github.com/woodenbirds/cs182/blob/master/imgs/sleeping origin.png" width="10%"> <img src="https://github.com/woodenbirds/cs182/blob/master/imgs/sunglass_origin.png" width="10%">
=>
<img src="https://github.com/woodenbirds/cs182/blob/master/imgs/shocking.png" width="10%"> <img src="https://github.com/woodenbirds/cs182/blob/master/imgs/sleeping.png" width="10%"> <img src="https://github.com/woodenbirds/cs182/blob/master/imgs/sunglasses.png" width="10%">

(shocked face emoji/sleeping face emoji/smiling face with sunglasses emoji)

## The fourth trial

In this trial, we try to add lora in the stable diffusion model. We train the model by using the mixed dataset of all the software types.

### Fine-tuning Instruction

We provide the corresponding code and dataset in [Fourth_Trial](Fourth_Trial)  folder. You can run files in [lora_finetune_attention_weight.ipynb](Fourth_Trial/lora_finetune_attention_weight.ipynb) to reproduce the model.

### Requirement

For the training requirement, we recommend using colab to run this file. You can upload the whole [Fourth_Trial](Fourth_Trial) folder to colab and run [lora_finetune_attention_weight.ipynb](Fourth_Trial/lora_finetune_attention_weight.ipynb). In order to save data and visualize it, we recommend that you use wandb in the code to synchronize data. For this, you may need to register a relevant account. 

### Training

During the training, you just need to follow the instruction in the [lora_finetune_attention_weight.ipynb](Fourth_Trial/lora_finetune_attention_weight.ipynb).

### Dataset

We used [Full Emoji Image Dataset](https://www.kaggle.com/datasets/subinium/emojiimage-dataset) on kaggle as our training and validation dataset. We use [test.ipynb](test.ipynb) to preprocess the data into the form we need, which are stored in the "Fourth Trial" folder. You can also try to process them by yourself. The data ratio is: train_set : validation_set : test_set = 100 : 1 : 20. For this trail, because we choose to train all the type together, the prompt of each image is in the form of '(description) + emoji in (typical) style'. You can find them in [data](Fourth_Trial/data).

## The fifth trial

In this trial, we try to add lora in the stable diffusion model. We want to find out if the mixed dataset is better or the single dataset is better. First, we train the model by using the single dataset of 'Apple' types. Next, we train the model by using the mixed dataset.

### Fine-tuning Instruction

We provide the corresponding code and dataset in [Fifth_Trial](Fifth_Trial)  folder. You can run files in [lora_finetune_attention_weight.ipynb](Fifth_Trial/lora_finetune_attention_weight.ipynb) to reproduce the model.

### Requirement

For the training requirement, we recommend using colab to run this file. You can upload the whole [Fifth Trial](Fifth Trial) folder to colab and run [lora_finetune_attention_weight.ipynb](Fifth_Trial/lora_finetune_attention_weight.ipynb). In order to save data and visualize it, we recommend that you use wandb in the code to synchronize data. For this, you may need to register a relevant account. 

### Training

During the training, you just need to follow the instruction in the [lora_finetune_attention_weight.ipynb](Fifth_Trial/lora_finetune_attention_weight.ipynb).

### Dataset

We used [Full Emoji Image Dataset](https://www.kaggle.com/datasets/subinium/emojiimage-dataset) on kaggle as our training and validation dataset. We use [test.ipynb](test.ipynb) to preprocess the data into the form we need, which are stored in the "Fourth Trial" folder. You can also try to process them by yourself. For the mixed dataset, the prompt of each image is in the form of '(description) + emoji in <sx> style', where sx = [s0, s1, s2, ..., s6] corresponding to ['Apple', 'Google', 'Facebook', 'Samsung', 'Windows', 'Twitter','JoyPixels']. The data ratio is: train_set : validation_set : test_set = 100 : 1 : 20. You can find them in [data](Fifth_Trial/data).For the single dataset, the prompt of each image is in the form of '(desciption) + emoji'. The data ratio is: train_set : validation_set : test_set = 100 : 1 : 20. You can find them in [Apple_style_data](Fifth_Trial/Apple_style_data).

## Final Result

We choose model with rank = 64, learning rate = 1e-4, weight decay = 0.01 as the final fine-tuned model. For the model, we add Lora to every blocks in Unet and add Lora on (q,k,v,o) matrix. The evaluation methods can be seen in both the last part of [lora_finetune_attention_weight.ipynb](Fifth_Trial/lora_finetune_attention_weight.ipynb) and [evaluation_score.py](evaluation_score.py).

You can see the checkpoints of the fine-tuned model in [latest_model_checkpoint](latest_model_checkpoint).

