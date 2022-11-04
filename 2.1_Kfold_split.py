# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:41:55 2022

@author: L. W. Shang
"""

import os
import shutil
from sklearn.model_selection import KFold

def folder_empty(path):
    file_list = os.listdir(path)
    if file_list:
        shutil.rmtree(path)  
        os.mkdir(path) 

def move_img(save_train_path, ca_path, train_index):
    for index in train_index:
        img_list = os.listdir(ca_path)
        img_list.sort(key=lambda x: int(x[:-4]))
        save_im_path = os.path.join(save_train_path, img_list[index])
        ori_im_path = os.path.join(ca_path, img_list[index])
        shutil.copyfile(ori_im_path, save_im_path)



os_path = os.getcwd()
pre_process_img_path = os.path.join(os_path, '1_pre_process_img')
split_augoment_img_path = os.path.join(os_path, '2_split_augoment_img')
folder_empty(split_augoment_img_path)

kf = KFold(n_splits = 10, shuffle=False, random_state=None)

ca_path = os.path.join(pre_process_img_path,'ca')
ca_list = os.listdir(ca_path)
ca_iter = 0
for train_index, test_index in kf.split(ca_list):
    ca_iter += 1
    save_iter_path = os.path.join(split_augoment_img_path, str(ca_iter))
    os.mkdir(save_iter_path)
    save_ca_path = os.path.join(save_iter_path, 'ca')
    os.mkdir(save_ca_path)
    
    save_train_path = os.path.join(save_ca_path, 'train')
    os.mkdir(save_train_path)
    move_img(save_train_path, ca_path, train_index)
    
    save_test_path = os.path.join(save_ca_path, 'test')
    os.mkdir(save_test_path)
    move_img(save_test_path, ca_path, test_index)
    
    
no_path = os.path.join(pre_process_img_path,'no')
no_list = os.listdir(no_path)
no_iter = 0
for train_index, test_index in kf.split(no_list):
    no_iter += 1
    save_iter_path = os.path.join(split_augoment_img_path, str(no_iter))
    save_no_path = os.path.join(save_iter_path, 'no')
    os.mkdir(save_no_path)
    
    save_train_path = os.path.join(save_no_path, 'train')
    os.mkdir(save_train_path)
    move_img(save_train_path, no_path, train_index)
    
    save_test_path = os.path.join(save_no_path, 'test')
    os.mkdir(save_test_path)
    move_img(save_test_path, no_path, test_index)
        
    
    
    