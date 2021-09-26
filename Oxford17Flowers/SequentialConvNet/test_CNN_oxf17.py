#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 23:46:30 2021

@author: tpeng
"""

import torch as tch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import shutil
import os
# Device configuration
device = tch.device('cuda:0' if tch.cuda.is_available() else 'cpu'); 
if device.type=='cuda':
    print("Cuda is enabled")

num_classes=17

#%%

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential (
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(24, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential (
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(),            
            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential (
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer4 = nn.Sequential (
            nn.Conv2d(128, 256, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.classifier = nn.Sequential(
            nn.Linear(256 * 10 * 10, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)


        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


#%% load data
home_dir=os.path.expanduser("~")
data_dir=home_dir+'/MEGAsync/ML_practice/Oxford17Flower/SequentialConvNet/data/'
gth_dir=data_dir+'/trimaps/'
jpg_dir=data_dir+'/jpg/'

# load mat
# ground_truth (848 images)
imlist=loadmat(gth_dir+'/imlist.mat')
# data_split
datasplit=loadmat(data_dir+'/datasplits.mat')

# convert index to 
ind_test=[datasplit['tst1'][0].astype(int),datasplit['tst2'][0].astype(int),datasplit['tst3'][0].astype(int)]
ind_train=[datasplit['trn1'][0].astype(int),datasplit['trn2'][0].astype(int),datasplit['trn3'][0].astype(int)]
ind_val=[datasplit['val1'][0].astype(int),datasplit['val2'][0].astype(int),datasplit['val3'][0].astype(int)]

# labels (each group has 80 files)
def get_label(index):
    return np.ceil(index/80).astype(int)

def get_pic_fname(index,path_src):
    num_ind=len(index)
    fname_str=['']*num_ind
    for i in range(num_ind):
        ind_str='{:04d}'.format(index[i])
        fname_str[i]=path_src+'image_'+ind_str+'.jpg'
    return fname_str

def get_label_list(ind_list,path_src):
    label_list=['']*len(ind_list)
    fname_list=['']*len(ind_list)
    for i in range(len(ind_list)):
        label_list[i]=get_label(ind_list[i])
        fname_list[i]=get_pic_fname(ind_list[i],path_src)
    return label_list,fname_list
# create a tmp folder 
def mkdir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
mkdir(data_dir+'./tmp/')
tmp_train_dir=data_dir+'/tmp/train/'; mkdir(tmp_train_dir)
tmp_test_dir=data_dir+'/tmp/test/'; mkdir(tmp_test_dir)
tmp_val_dir=data_dir+'/tmp/val/'; mkdir(tmp_val_dir)

label_test,fname_test=get_label_list(ind_test,jpg_dir)
label_train,fname_train=get_label_list(ind_train,jpg_dir)
label_val,fname_val=get_label_list(ind_val,jpg_dir)

def sort_pic_groups(fname_list,path_str,label_list):
    num_subgroup=len(fname_list)
    for i in range(num_subgroup):
        path_subgroup=path_str+'g{:02d}'.format(i)+'/'
        mkdir(path_subgroup)
        print(path_subgroup)
        for j in range(len(fname_list[i])):
            path_class=path_subgroup+'c{:02d}'.format(label_list[i][j])
            mkdir(path_class)
            shutil.copy(fname_list[i][j],path_class)
s1=sort_pic_groups(fname_test,tmp_test_dir,label_test); del s1
s2=sort_pic_groups(fname_train,tmp_train_dir,label_train); del s2
s3=sort_pic_groups(fname_val,tmp_val_dir,label_val); del s3


#%%
# Hyper pyarameters
image_res=192
num_epochs = 15
num_classes = 17
batch_size = 50
learning_rate = 0.00125
lr_inflation=0.75 #assuming generality is obtained at the first round
bt_inflation=1.25
ep_inflation=0.6
#% define model

model = ConvNet(num_classes).to(device)
NumParams=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters in model =',NumParams)
#%

# Define transform
transform = transforms.Compose([
    transforms.Resize(image_res),
    transforms.CenterCrop(image_res),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
#% Training data



#% Train the model
num_group=3;
for ig in range(num_group):
    if (ig!=0):
        learning_rate=learning_rate*lr_inflation
        batch_size=int(batch_size*bt_inflation)
        num_epochs=int(num_epochs*ep_inflation)
    # define data loader
    def data_loader(train_data_path,val_data_path):
        train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
        train_loader = tch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
        val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=transform)
        val_loader  = tch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4) 
        
        train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
        train_loader = tch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
        val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=transform)
        val_loader  = tch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4) 
        
        train_class_names = train_data.classes
        val_class_names = val_data.classes
        
        return train_loader, val_loader, train_class_names, val_class_names

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    # optimizer = tch.optim.Adadelta(model.parameters(), lr=learning_rate)
    optimizer = tch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = tch.optim.SGD(model.parameters(), lr=learning_rate)
    group_str='{:02d}'.format(ig)
    print('Training group: '+group_str)
    [train_loader, test_loader, train_class_names, test_class_names]=data_loader(tmp_train_dir+'/g'+group_str+'/',tmp_test_dir+'/g'+group_str+'/')
    # display samples
    display_ind=(np.random.rand(5)*1300).astype(int)[0]
    images, labels = iter(train_loader).next()
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title('Sample images (randomly selected)')
        plt.show()
    imshow(torchvision.utils.make_grid(images[:4]))
    
    
    total_step = len(train_loader)
    LossesToPlot=np.zeros(num_epochs*total_step)
    j=0
    for epoch in range(num_epochs):
        print('# of epoch: '+str(epoch))
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LossesToPlot[j]=loss.item()
            j+=1
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    print('Final Loss = ',loss.item()) # Print final loss
    
    # Plot Loss as a function of Mini-Batch
    plt.semilogy(LossesToPlot)
    plt.xlabel('Mini-Batch number')
    plt.ylabel('Loss')
    plt.show()
    
    #% Training data
    with tch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = tch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        print('Percent of training images misclassified: {} %'.format(100-100 * correct / total))
    #% Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with tch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = tch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        print('Percent of testing images misclassified: {} %'.format(100-100 * correct / total))
