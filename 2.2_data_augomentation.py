# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:58:46 2022

@author: L. W. Shang
"""

import os
import cv2
import numpy

img_augoment_num = 5000

def img_flip(augoment_ca_path):
    img_list = os.listdir(augoment_ca_path)
    for img_name in img_list:
        img_path = os.path.join(augoment_ca_path, img_name)
        img = cv2.imread(img_path)
        img_flip0 = cv2.flip(img,0)
        save_path = os.path.join(augoment_ca_path, img_name.replace('.jpg', '') + '_flip0.jpg')
        cv2.imwrite(save_path, img_flip0)
    
    img_list = os.listdir(augoment_ca_path)
    for img_name in img_list:
        img_path = os.path.join(augoment_ca_path, img_name)
        img = cv2.imread(img_path)
        img_flip1 = cv2.flip(img,0)
        save_path = os.path.join(augoment_ca_path, img_name.replace('.jpg', '') + '_flip1.jpg')
        cv2.imwrite(save_path, img_flip1)


def img_rotate(augoment_ca_path):
    img_list = os.listdir(augoment_ca_path)
    for img_name in img_list:
        img_path = os.path.join(augoment_ca_path, img_name)
        img = cv2.imread(img_path)
        
        rot_mat45 =  cv2.getRotationMatrix2D((100,100), -45, 1)
        img_ro45 = cv2.warpAffine(img, rot_mat45, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(augoment_ca_path, img_name.replace('.jpg', '') + '_ro45.jpg'), img_ro45)
        
        rot_mat135 =  cv2.getRotationMatrix2D((100,100), -135, 1)
        img_ro135 = cv2.warpAffine(img, rot_mat135, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(augoment_ca_path, img_name.replace('.jpg', '') + '_ro135.jpg'), img_ro135)
        
        rot_mat225 =  cv2.getRotationMatrix2D((100,100), -225, 1)
        img_ro225 = cv2.warpAffine(img, rot_mat225, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(augoment_ca_path, img_name.replace('.jpg', '') + '_ro225.jpg'), img_ro225)
        
        rot_mat315 =  cv2.getRotationMatrix2D((100,100), -315, 1)
        img_ro315 = cv2.warpAffine(img, rot_mat315, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(augoment_ca_path, img_name.replace('.jpg', '') + '_ro315.jpg'), img_ro315)

def img_noise(augoment_ca_path):
    img_list = os.listdir(augoment_ca_path)
    while len(img_list) < 5000:
        rand_index = int(numpy.random.random()* len(img_list))
        img_name = img_list[rand_index]
        img_path = os.path.join(augoment_ca_path, img_name)
        img = cv2.imread(img_path)
        gauss = numpy.random.normal(0,25,(img.shape[1],img.shape[1],3))
        noisy_img = img + gauss
        noisy_img = numpy.clip(noisy_img,a_min=0,a_max=255)
        save_path = os.path.join(augoment_ca_path, img_name.replace('.jpg', '') + '_noise.jpg')
        cv2.imwrite(save_path, noisy_img)
        img_list = os.listdir(augoment_ca_path)


os_path = os.getcwd()
split_augoment_img_path = os.path.join(os_path, '2_split_augoment_img')

split_list = os.listdir(split_augoment_img_path)
for split_name in split_list:
    print(split_name)
    split_path = os.path.join(split_augoment_img_path, split_name)
    
    augoment_ca_path = os.path.join(split_path,'ca','train')
    augoment_no_path = os.path.join(split_path,'no','train')
    
    img_flip(augoment_ca_path)
    img_flip(augoment_no_path)
    
    img_rotate(augoment_ca_path)
    img_rotate(augoment_no_path)
    
    img_noise(augoment_ca_path)
    img_noise(augoment_no_path)

