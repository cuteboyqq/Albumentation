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
    if not os.path.exists("aug_images"):
        os.makedirs("aug_images")
    img_name = str(count)+ "_" + type_+ ".png"
    plt.savefig("aug_images/" + img_name)

img_list = glob.glob(os.path.join("images","*.jpg"))
count = 1
for img in img_list:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    visualize(image,count,type_="ori")
    
    transform = A.Compose([
        A.Perspective(),
        #A.RandomCrop(360, 640),
        A.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=True, p=0.5)
        #A.RandomSunFlare (flare_roi=(0.0, 0.0, 0.5, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=600, src_color=(255, 255, 255), always_apply=True, p=0.5)
        #A.RandomSnow (snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=True, p=0.5)
        #A.RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=2, num_shadows_upper=4, shadow_dimension=7, always_apply=True, p=0.5)
        #A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=1,alpha_coef=0.06, always_apply=True, p=0.5),
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
    visualize(transformed['image'],count,type_="gamma")     
    count+=1
