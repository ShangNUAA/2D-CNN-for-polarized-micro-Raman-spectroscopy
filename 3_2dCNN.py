# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:37:07 2020

@author: Shang@nuaa.edu.cn
"""


#引用环境变量
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy
import visdom



#定义超参数
torch.set_default_tensor_type(torch.FloatTensor)#定义全局新建的tensortype为float
batchsize = 200        #定义一个全局变量batchsize
learningrate = 0.00005    #定义一个全局变量learningrate
epoches = 100             #定义一个全局变量epoches
cancer_label_dir = "ca"  
normal_label_dir = "no" 




"""-------------------------定义三个需要用到的函数---------------------------"""
def label_renumber(label):#用于将label转化为数值1和2，便于计算比较
    if label == 'ca': 
        number = numpy.array([0]) #如果label_cahr=‘cancer’，则将1添加到label_number中        
    if label == 'no':
        number = numpy.array([1])
    number = torch.FloatTensor(number) #转化为float形的tensor
    return number

def output_renumber(outputs):#用于将output的结果以1.5为阈值赋值为1和2，便于计算正确率
    output_renumber = []
    for output_number in outputs:
        if output_number < 0.5:
            number = numpy.array([0])
            output_renumber.append(number)
        if output_number >= 0.5:
            number = numpy.array([1])
            output_renumber.append(number)  
    output_renumber = torch.FloatTensor(output_renumber)#结果转化为Double形的tensor 
    return output_renumber


def statistical_accuracy(labels, outputs):#用于依次比较labels和outputs里面的值，统计正确率
    accuracy = 0
    for i in range(len(labels)):
        if labels[i] == outputs[i]:
            accuracy +=1
    return accuracy



"""---------------------------------第一步----------------------------------"""
"""-----------------------定义数据集和数据加载方式---------------------------"""
transform = T.Compose([
    T.Resize(100),
    T.CenterCrop(100),
    T.ToTensor(),
    T.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
    ])#定义读取图像的尺寸为100*100，并转换为tensor，归一化


class my_data(Dataset):#定义数据集my_data类
    def __init__(self, root_dir, label_dir, train_test, transforms=None):#传入的数据为root_dir(根目录)和label_dir（标签目录）
        self.root_dir = root_dir#定义数据集中的根目录变量
        self.label_dir = label_dir#定义数据集中的标签目录变量
        self.path = os.path.join(self.root_dir, self.label_dir, train_test)#将根目录与标签目录连接
        self.img_path = os. listdir(self.path)#获得该目录下所有文件的列表
        self.transforms = transforms
        
    def __getitem__(self, index):#按顺序依次读取文件，传入的数据index为当前文件的指针
        img_name = self.img_path[index]#从文件列表中获得index当前指向的文件的名称并保存在img_name中
        img_item_path = os.path.join(self.path, img_name)#将当前文件的存放目录与文件名称连接
        img = Image.open(img_item_path)#以图像的形式打开该文件
        if self.transforms:
            img = self.transforms(img)#图像大小缩放为100*100
        label = label_renumber(self.label_dir)#当前文件的标签
        return img, label#返回打开的图片和图片对应的指针
    
    def __len__(self):#返回文件的数量（文件列表的长度）
        return len(self.img_path)
  
  
"""--------------------------------第二步-----------------------------------"""
"""---------------------------定义神经网络模型-------------------------------"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=10, stride=2)
        self.batchnormal1 = nn.BatchNorm2d(10)#批标准化层1
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#池化层1
        
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=2)
        self.batchnormal2 = nn.BatchNorm2d(20)#批标准化层2
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)#池化层2       
        
        self.fc1 = nn.Linear(20*5*5, 20)#32个通道，图像尺寸为5*5
        self.batchnormal3 = nn.BatchNorm1d(20)#批标准化层2
        self.dropout1 = nn.Dropout(0.5)#dropout1
        
        self.fc2 = nn.Linear(20,1)#最终一个通道输出结果
        
    def forward(self, x):
        x = F.relu(self.batchnormal1(self.conv1(x)))
        x = self.max_pool1(x)
        x = F.relu(self.batchnormal2(self.conv2(x)))
        x = self.max_pool2(x)

        x = x.view(x.size()[0],-1)
        x = F.relu(self.batchnormal3(self.fc1(x)))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x



"""---------------------------------第四步----------------------------------"""
"""----------------主函数（训练、反向传递、优化及作训练曲线）-----------------"""
      
os_path = os.getcwd()#获取当前路径
split_augoment_img_path = os.path.join(os_path, '2_split_augoment_img')
split_list = os.listdir(split_augoment_img_path)
result_path = os.path.join(os_path, '3_train_result')


vis = visdom.Visdom()#定义一个Visdom类的实例vis，用于作图


for split_name in  split_list:
    net = Net()
    criterion = nn.BCELoss() #用BCE作为损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr = learningrate) #用Adam作为优化器
    net = net.cuda()#将net模型导入gpu
    criterion = criterion.cuda()#将损失函数导入gpu

    split_path = os.path.join(split_augoment_img_path, split_name)
    training_cancer_datasets = my_data(split_path, cancer_label_dir, 'train', transforms = transform)#获得cancer数据集
    training_normal_datasets = my_data(split_path, normal_label_dir, 'train', transforms = transform)#获得normal数据集
    train_datasets = training_cancer_datasets + training_normal_datasets#将cancer和normal数据集相加，即为训练集
    train_loader = DataLoader(dataset = train_datasets,#使用DataLoader类加载训练集
                          batch_size = batchsize,
                          shuffle = True, #打乱顺序
                          num_workers = 0) #传递数据的CPU核心数

    test_cancer_datasets = my_data(split_path, cancer_label_dir, 'test', transforms = transform)#获得cancer数据集
    test_normal_datasets = my_data(split_path, normal_label_dir, 'test', transforms = transform)#获得normal数据集
    test_datasets = test_cancer_datasets + test_normal_datasets#将cancer和normal数据集相加，即为训练集
    test_loader = DataLoader(dataset = test_datasets,#使用DataLoader类加载训练集
                          batch_size = batchsize,
                          shuffle = None, #打乱顺序
                          num_workers = 0) #传递数据的CPU核心数
    
    iter_count = 0#记录当前的interations，用于作图
    for epoch in range(epoches): #按周期循环
        for i, data in enumerate(train_loader,0): #按batchsize循环，将迭代对象组合成一个索引
            inputs, labels = data #将batch中的数据、标签分别保存到inputs和labels中       
            optimizer.zero_grad()#优化器的梯度清零
            inputs = inputs.cuda()#将inputs导入gpu，得到的outputs也在gpu中
            outputs = net(inputs)#输入inputs，得到outputs
            labels = labels.cuda()
            loss = criterion(outputs, labels)#计算损失函数
            loss.backward()#损失函数反向传递
            optimizer.step()#对网络参数进行优化
            
            outputs = output_renumber(outputs)#将outputs按阈值编码为1和2
            outputs = outputs.cuda()
        
            accuracy = statistical_accuracy(labels, outputs)#计算正确率
            accuracy = accuracy / len(labels) * 100
        
    
            val_inputs, val_labels = next(iter(train_loader)) #将batch中的数据、标签分别保存到val_inputs和val_labels中
            val_inputs = val_inputs.cuda()#（可选）将val_inputs导入gpu，则得到的val_outputs也在gpu中
            val_outputs = net(val_inputs)#输入val_inputs到神经网络，得到val_outputs
            val_labels = val_labels.cuda()#（可选）将标签导入gpu
            val_loss = criterion(val_outputs, val_labels)#计算损失函数
   
            val_outputs = output_renumber(val_outputs)
            val_outputs = val_outputs.cuda()     
            val_accuracy = statistical_accuracy(val_labels, val_outputs)
            val_accuracy = val_accuracy / len(val_labels) * 100
            
            
            test_inputs, test_labels = next(iter(test_loader)) #将batch中的数据、标签分别保存到val_inputs和val_labels中
            test_inputs = test_inputs.cuda()#（可选）将val_inputs导入gpu，则得到的val_outputs也在gpu中
            test_outputs = net(test_inputs)#输入val_inputs到神经网络，得到val_outputs
            test_labels = test_labels.cuda()#（可选）将标签导入gpu
            test_loss = criterion(test_outputs, test_labels)#计算损失函数
   
            test_outputs = output_renumber(test_outputs)
            test_outputs = test_outputs.cuda()     
            test_accuracy = statistical_accuracy(test_labels, test_outputs)
            test_accuracy = test_accuracy / len(test_labels) * 100
            

            print("epoch:", epoch, "iter:", i, "tra_loss:", round(loss.item(),2),"train_acc:",round(accuracy,2),
                  "val_loss:", round(val_loss.item(),2), 'val_acc', round(val_accuracy,2),
                  "test_loss:", round(test_loss.item(),2), 'test_acc', round(test_accuracy,2)
                  )#显示当前的周期，迭代次数和损失函数的值


            #以下调用visdom类对训练曲线进行输出 -------------------------------------------- 
            iter_count += 1
            if iter_count == 1:
                vis.line(Y = numpy.column_stack((numpy.array([loss.item()]), numpy.array([val_loss.item()]))), 
                     X = numpy.column_stack((numpy.array([iter_count]), numpy.array([iter_count]))), 
                     win = 'loss', update = 'replace', 
                     opts = dict(legned = ['Train_loss','val_loss'], title = 'loss'))
                vis.line(Y = numpy.column_stack((numpy.array([accuracy]), numpy.array([val_accuracy]))), 
                     X = numpy.column_stack((numpy.array([iter_count]), numpy.array([iter_count]))), 
                     win ='accy', update = 'replace', 
                     opts = dict(legned = ['Train_accuracy','val_loss'], title = 'accuracy'))
            
            else:
                vis.line(Y = numpy.column_stack((numpy.array([loss.item()]), numpy.array([val_loss.item()]))), 
                     X = numpy.column_stack((numpy.array([iter_count]), numpy.array([iter_count]))), 
                     win='loss', update = 'append')
                vis.line(Y = numpy.column_stack((numpy.array([accuracy]), numpy.array([val_accuracy]))), 
                     X = numpy.column_stack((numpy.array([iter_count]), numpy.array([iter_count]))), 
                     win='accy', update = 'append')
         
    model_path = os.path.join(result_path, split_name + '.pt')
    torch.save(net, model_path)



