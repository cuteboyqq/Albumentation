#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 13:19:56 2023

@author: jnr_loganvo
"""
import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomSnow(),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("images/2.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#cv2.imshow('My Image', image)

# 按下任意鍵則關閉所有視窗
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]

#cv2.imshow('My Image', transformed_image)

# 按下任意鍵則關閉所有視窗
#cv2.waitKey(0)
#cv2.destroyAllWindows()