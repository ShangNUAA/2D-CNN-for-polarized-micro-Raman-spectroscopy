# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 21:12:12 2022

@author: L. W. Shang
"""

import os
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import cv2
import matplotlib.pyplot as plt
kf = KFold(n_splits = 3, shuffle=True, random_state=0)

ks = 4


"""----------------------------------函数定义-------------------------------"""
#输入文件路径，文件列表，返回所有文件的label列表

def split_index(ca_list):
    for ca_train_index, ca_val_index in kf.split(ca_list):
        x=1
    return ca_train_index, ca_val_index

def get_data_label(ca_train_val_path, ca_list, ca_train_index, label):
    data_list = []
    label_list = []
    for index in ca_train_index:
        img_path = os.path.join(ca_train_val_path, ca_list[index])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([img], [0, 1], None, [50, 50], [0.0, 255.0, 0.0, 255.0])
        data_list.append(((hist/255).flatten()))
        if label == 'ca':
            label_list.append(0)
        elif label == 'no':
            label_list.append(1)
    return data_list, label_list


def get_train_val_data_label(ca_train_val_path, no_train_val_path):
    ca_list = os.listdir(ca_train_val_path)
    no_list = os.listdir(no_train_val_path)
    ca_train_index, ca_val_index = split_index(ca_list)
    no_train_index, no_val_index = split_index(ca_list)
    
    ca_train_data, ca_train_label = get_data_label(ca_train_val_path, ca_list, ca_train_index, 'ca')
    no_train_data, no_train_label= get_data_label(no_train_val_path, no_list, no_train_index, 'no')
    
    ca_val_data, ca_val_label = get_data_label(ca_train_val_path, ca_list, ca_val_index, 'ca')
    no_val_data, no_val_label = get_data_label(no_train_val_path, no_list, no_val_index, 'no')
    
    train_data = ca_train_data + no_train_data
    val_data = ca_val_data + no_val_data
    
    train_label = ca_train_label + no_train_label
    val_label = ca_val_label + no_val_label
    
    return train_data, train_label, val_data, val_label

def get_test_data_label(ca_test_path, no_test_path):
    ca_test_list = os.listdir(ca_test_path)
    no_test_list = os.listdir(no_test_path)
    
    data_list = []
    label_list = []
    for ca_img_name in ca_test_list:
        img_path = os.path.join(ca_test_path, ca_img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([img], [0, 1], None, [50, 50], [0.0, 255.0, 0.0, 255.0])
        data_list.append(((hist/255).flatten()))
        label_list.append(0)
        
    for no_img_name in no_test_list:
        img_path = os.path.join(no_test_path, no_img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([img], [0, 1], None, [50, 50], [0.0, 255.0, 0.0, 255.0])
        data_list.append(((hist/255).flatten()))
        label_list.append(1)
    
    return data_list, label_list

def cal_mean_std(total_train_matrix):
    mean_matrix = numpy.mean(total_train_matrix, axis=0)
    std_matrix = numpy.std(total_train_matrix, axis=0)
    return mean_matrix, std_matrix
    
def statistical_accuracy(labels, outputs):#用于依次比较labels和outputs里面的值，统计正确率
    matrix = numpy.zeros((2,2))
    for i in range(len(labels)):
        matrix[int(labels[i]),int(outputs[i])] += 1
    return matrix

def matrix_trans(train_matrix):
    train_matrix[0,:] /= (train_matrix[0,0] + train_matrix[0,1] ) /100
    train_matrix[1,:] /= (train_matrix[1,0] + train_matrix[1,1] ) /100
    return train_matrix

def draw_fusion_matrix(confusion, mse, color):
    plt.imshow(confusion, cmap= color)
    indices = range(len(confusion))
    classes = list(range(len(confusion[0])))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('guess')
    plt.ylabel('fact')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            if first_index == second_index:
                plt.text(second_index-0.26, first_index-0.12, str(round(confusion[first_index][second_index],2)), size = 20, color = 'white')
                plt.text(second_index-0.12, first_index+0.08, '±', size = 18, color = 'white')
                plt.text(second_index-0.22, first_index+0.28, str(round(mse[first_index][second_index],2)), size = 20, color = 'white')
            else:
                plt.text(second_index-0.26, first_index-0.12, str(round(confusion[first_index][second_index],2)), size = 20)
                plt.text(second_index-0.12, first_index+0.08, '±', size = 18)
                plt.text(second_index-0.22, first_index+0.28, str(round(mse[first_index][second_index],2)), size = 20)
    plt.show()

"""----------------------------------主函数---------------------------------"""
print('计算开始')
os_path = os.getcwd()
split_augoment_img_path = os.path.join(os_path, '2_split_augoment_img')
split_list = os.listdir(split_augoment_img_path)
result_path = os.path.join(os_path, '4_knn_result')
total_train_matrix = []
total_val_matrix = []
total_test_matrix = []
for split_name in  split_list:
    print('当前处理第', split_name, '折，开始加载数据')
    split_path = os.path.join(split_augoment_img_path, split_name)
    ca_path = os.path.join(split_path, 'ca')
    no_path = os.path.join(split_path, 'no')
    ca_train_val_path = os.path.join(ca_path, 'train')
    no_train_val_path = os.path.join(no_path, 'train')
    
    ca_test_path = os.path.join(ca_path, 'test')
    no_test_path = os.path.join(no_path, 'test')
    
    train_data, train_label, val_data, val_label = get_train_val_data_label(ca_train_val_path, no_train_val_path)
    test_data, test_label = get_test_data_label(ca_test_path, no_test_path)
    print('所有数据和标签列表加载成功！')
    print('正在训练KNN模型')
    knn_model = KNeighborsClassifier(n_neighbors=ks)
    knn_model.fit(train_data, train_label)
    print('KNN模型训练完成！')
    print('开始进行KNN预测：')
    print('正在预测。。。。。。')   
    train_result = knn_model.predict(train_data)
    val_result = knn_model.predict(val_data)  
    test_result = knn_model.predict(test_data)
    
    train_matrix = statistical_accuracy(train_label, train_result)
    val_matrix = statistical_accuracy(val_label, val_result)
    test_matrix = statistical_accuracy(test_label, test_result)
    
    train_matrix = matrix_trans(train_matrix)
    val_matrix = matrix_trans(val_matrix)
    test_matrix = matrix_trans(test_matrix)
    
    total_train_matrix.append(train_matrix)
    total_val_matrix.append(val_matrix)
    total_test_matrix.append(test_matrix)
   
total_train_matrix = numpy.array(total_train_matrix)   
total_val_matrix = numpy.array(total_val_matrix)   
total_test_matrix = numpy.array(total_test_matrix)   

train_save_path = os.path.join(result_path, 'train.npy')
val_save_path = os.path.join(result_path, 'val.npy')
test_save_path = os.path.join(result_path, 'test.npy')

numpy.save(train_save_path, total_train_matrix)
numpy.save(val_save_path, total_val_matrix)
numpy.save(test_save_path, total_test_matrix)

train_confusion, train_std = cal_mean_std(total_train_matrix)
val_confusion, val_std = cal_mean_std(total_val_matrix)
test_confusion, test_std = cal_mean_std(total_test_matrix)

draw_fusion_matrix(train_confusion, train_std, 'Blues')
draw_fusion_matrix(val_confusion, val_std, 'Greens')
draw_fusion_matrix(test_confusion, test_std, 'Oranges')