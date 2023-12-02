### This file includes functions to evaluate the selected diffusion model ###
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image
import os
import numpy as np
import torch.nn as nn
from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

device = 'cuda'
weight_dtype = torch.float16

# load the FID score function
fid = FrechetInceptionDistance(normalize=True)
# load the clip score function
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
# first load our learnt unet to the stable diffusion pipline 
# images = pipline(prompts,....) then pass images into the func
# prompts = ["...","...",...]
def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

# image process func before calculating FID score
def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))


# TODO: create prompts with ["","",...] from our test dataset
# read test_emoji.csv; and put all the prompts to test_prompts
# TODO test_csv_path: the path to test_emoji.csv
test_csv_path = "" 
test_prompts = []

# our based pretrained model is "runwayml/stable-diffusion-v1-5"
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
# load our pretained model
pipeline = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,weight_dtype = torch.float16).to(device)
# output_dir: the directory where we save the finetuned weight of lora
pipeline.unet.load_attn_procs(output_dir)
generator = torch.Generator()
generator = generator.manual_seed(seed)
test_images = []
for i  in range(len(test_prompts)):
    test_images.append(pipeline(test_prompts[i],num_inference_steps=30,generator=generator).images[0])
# calcualate CLIP score
CLIP_score = calculate_clip_score(test_images,test_prompts)
print(f"CLIP Score : {CLIP_score}")
# write CLIP_score to wandb
wandb.log({"CLIP Score":CLIP_score})


# calculate FID score
# load the funcitons used to calculate FID score
# TODO dataset_path: the path to the test dataset
dataset_path = "" 
image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])
real_test_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
real_test_images = torch.cat([preprocess_image(image) for image in real_test_images])
test_images = torch.tensor(test_images)
test_images = test_images.permute(0,3,1,2)
fid.update(real_test_images, real=True)
fid.update(test_images, real=False)
print(f"FID: {float(fid.compute())}")
# write FID_score to wandb
wandb.log({"FID Score":float(fid.compute())})