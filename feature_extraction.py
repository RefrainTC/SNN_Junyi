'''
Loads ANN model and extracts feature vectors at the hidden layer and also plots histogram

Loads ANN model and copies the weights to an SNN model. 
'''


import torch
import torch.nn as nn
from spiking_layer_ours import *
from Models import modelpool
from Preprocess import datapool
#from torchvision.models.feature_extraction import create_feature_extractor
import os
import argparse
from funcs import *
import numpy as np
import time
import sys

import calc_th_with_c as ft

parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')

#parser.add_argument('--t', default=1, type=int, help='T Latency length (Simulation time-steps)')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'imagenet','tiny-imagenet','fashion'])
parser.add_argument('--model', default='vgg16', type=str, help='Model name',
                    choices=['cifarnet','small','vgg16', 'resnet18', 'resnet20',
                             'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed',
                             'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--checkpoint', default='./saved_models', type=str, help='Directory for saving models')

parser.add_argument('--iter', default=200, type=int, help='Number of iterations for finding th values')
parser.add_argument('--samples', default=10000, type=int, help='Number of iterations for finding th values')
args = parser.parse_args()
args.mid = f'{args.dataset}_{args.model}'
savename = os.path.join(args.checkpoint, args.mid)+"_new"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size =128



# to extract activation values
sample = 0
n_steps = 1
thresholds_13= np.zeros(n_steps)
def extract_features(L=2):
    train_loader, test_loader = datapool(args.dataset, batch_size,2)
    print("L=%d, n_steps = %d "%(L,n_steps))
    with torch.no_grad():
        #model.to(device)
        #model.eval()
        features = []
        Images = []
        count = 0
        if L==1 and n_steps==1:
            for images, labels in train_loader:
                Images.append(images)
                outputs = model(images.to(device),L=L)#),savename=savename)
                count += images.shape[0]
                features.append(outputs.detach().cpu())
                if count >= args.samples:
                    break        
            Images = torch.cat(Images).numpy()
            np.save('%s_Images.npy'%(args.dataset),Images)
            print(Images.shape)
        else:
            Images = np.load('%s_Images.npy'%(args.dataset))
            for i in range(Images.shape[0]//batch_size):
                images = torch.Tensor(Images[i*batch_size:(i+1)*batch_size])
                outputs = model(images.to(device),L=L)
                count += images.shape[0]
                features.append(outputs.detach().cpu())
                if count>=args.samples:
                     break

        features = torch.cat(features).flatten().numpy()
        features = features[features>0.0]
        features.sort()
        features = features.astype(np.longdouble)
        print("total features before removing",features.shape)
        #print(features[0:10])
        th_pos = ft.intl2(features,n_steps)
        
        for i in range(args.iter):
            th_pos = ft.optimize1(features,th_pos)
            # print("Error after iteration ",i," is ",ft.error(features, th_pos))
        
        thresholds_pos_all[L-1] = np.array(th_pos)
        thrs_in_list, thrs_out_list = ft.thrs_in_out(features, th_pos)
        thresholds_all[L-1,0:n_steps] = np.array(thrs_out_list)
        thresholds_all[L-1,n_steps:] = np.array(thrs_in_list)
        
        
        
        
        
        
        

model = modelpool(args.model, args.dataset)
criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load(savename + '.pth'))
model.to(device)
num_relu = str(model).count('ReLU')
print(num_relu)

thresholds_all = np.zeros((num_relu,n_steps*2))
thresholds_pos_all = np.zeros((num_relu,n_steps*2))

for i in range(num_relu):#num_relu):
    extract_features(L=i+1)
np.save('%s_threshold_all_noaug%d.npy'%(savename,n_steps),thresholds_all)
np.save('%s_threshold_pos_all_noaug%d.npy'%(savename,n_steps),thresholds_pos_all)

