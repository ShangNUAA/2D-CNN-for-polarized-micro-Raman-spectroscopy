# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:17:20 2022

@author: L. W. Shang
"""

import os
import shutil
import cv2


def folder_empty(path):
    file_list = os.listdir(path)
    if file_list:
        shutil.rmtree(path)  
        os.mkdir(path) 


os_path = os.getcwd()
ori_img_path = os.path.join(os_path, '0_ori_img')
pre_process_img_path = os.path.join(os_path, '1_pre_process_img')
folder_empty(pre_process_img_path)


ori_ca_path = os.path.join(ori_img_path, 'ca')
ori_ca_list = os.listdir(ori_ca_path)
save_ca_path = os.path.join(pre_process_img_path, 'ca')
os.mkdir(save_ca_path)

for img_name in ori_ca_list:
    img_path = os.path.join(ori_ca_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (500, 500))
    crop_img = img[19:457, 51:447, :]
    crop_img = cv2.resize(crop_img, (200, 200))
    img_save_path = os.path.join(save_ca_path, img_name)
    cv2.imwrite(img_save_path, crop_img)


ori_no_path = os.path.join(ori_img_path, 'no')
ori_no_list = os.listdir(ori_no_path)
save_no_path = os.path.join(pre_process_img_path, 'no')
os.mkdir(save_no_path)

for img_name in ori_no_list:
    img_path = os.path.join(ori_no_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (500, 500))
    crop_img = img[19:457, 51:447, :]
    crop_img = cv2.resize(crop_img, (200, 200))
    img_save_path = os.path.join(save_no_path, img_name)
    cv2.imwrite(img_save_path, crop_img)






"""
for patient_iter in range(1,11):
    patient_save_path = os.path.join(pre_process_img_path, str(patient_iter), 'ca')
    os.makedirs(patient_save_path)
    
    for img_iter in range((patient_iter-1)*20+1, patient_iter*20+1):
        img_path = os.path.join(ori_img_path, 'ca', str(img_iter)+'.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (500, 500))
            crop_img = img[19:457, 51:447, :]
            img_save_path = os.path.join(patient_save_path, str(img_iter)+'.jpg')
            cv2.imwrite(img_save_path, crop_img)
            
for normal_iter in range(1,11):
    normal_save_path = os.path.join(pre_process_img_path, str(normal_iter), 'no')
    os.makedirs(normal_save_path)
    
    for img_iter in range((normal_iter-1)*10+1, normal_iter*10+1):
        img_path = os.path.join(ori_img_path, 'no', str(img_iter)+'.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (500, 500))
            crop_img = img[19:457, 51:447, :]
            img_save_path = os.path.join(normal_save_path, str(img_iter)+'.jpg')
            cv2.imwrite(img_save_path, crop_img)
"""    