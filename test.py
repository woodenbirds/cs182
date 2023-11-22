import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image
import io
import os
import glob
import base64
import cv2

emoji = pd.read_csv('./full_emoji.csv')
emoji.head()

base64_decoded = base64.b64decode(emoji['Apple'][0].split(',')[-1])
image = Image.open(io.BytesIO(base64_decoded)).convert('RGBA')

plt.imshow(image)
breakpoint()