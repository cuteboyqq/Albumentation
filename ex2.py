#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 13:30:47 2023

@author: jnr_loganvo
"""
#code is from https://albumentations.ai/docs/examples/serialization/
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import glob
import os
# study https://albumentations.ai/docs/getting_started/image_augmentation/

def visualize(image,count,type_):
    plt.figure(figsize=(150,90))
    plt.axis('off')
    plt.imshow(image)
    if not os.path.exists(os.path.join("aug_images",type_)):
        os.makedirs(os.path.join("aug_images",type_))
    img_name = str(count)+ "_" + type_+ ".png"
    plt.savefig("aug_images/"+ type_ + "/" + img_name)

img_list = glob.glob(os.path.join("images","*.jpg"))
count = 1
for img in img_list:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    visualize(image,count,type_="ori")
    
    transform = A.Compose([
        #A.Perspective(),
        #A.RandomCrop(360, 640),
        #A.RandomGravel (gravel_roi=(0.1, 0.4, 0.9, 0.9), number_of_patches=2, always_apply=False, p=0.5)
        #A.ISONoise (color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=True, p=0.5)
        #A.PixelDropout (dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, always_apply=True, p=0.5)
        #A.RandomBrightness (limit=0.7, always_apply=True, p=0.5)
        #A.RandomBrightnessContrast (brightness_limit=0.05, contrast_limit=0.05, brightness_by_max=False, always_apply=True, p=0.5) 
        #A.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=True, p=0.5)
        A.RandomSunFlare (flare_roi=(0.5, 0.2, 0.6, 0.3), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=450, src_color=(255, 255, 255), always_apply=True, p=0.5)
        #A.RandomSnow (snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=True, p=0.5)
        #A.RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=2, num_shadows_upper=4, shadow_dimension=7, always_apply=True, p=0.5)
        #A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1.0,alpha_coef=0.10, always_apply=True, p=0.5),
        #A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=True, p=0.5)
        #A.OneOf([
        #    #A.RGBShift(), 
        #    #A.HueSaturationValue(),
        #    A.RandomFog(fog_coef_lower=0.6, fog_coef_upper=1,alpha_coef=0.50, always_apply=True, p=0.5),
        #]),
    ])
    
    print(transform)
    
    
    random.seed(42)
    np.random.seed(42)
    transformed = transform(image=image)
    visualize(transformed['image'],count,type_="sunflare")     
    count+=1
