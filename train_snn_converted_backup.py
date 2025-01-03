import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from spiking_layer_ours import *
from Models import modelpool
from Preprocess import datapool
from torchvision.models.feature_extraction import create_feature_extractor
import os
import argparse
from funcs import *
import numpy as np
import time
import sys
from utils_my import *
import calc_th_with_c as ft

parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')

parser.add_argument('--t', default=300, type=int, help='T Latency length (Simulation time-steps)')
parser.add_argument('--TET', default=0, type=int, help='TET')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                    choices=['cifar10', 'cifar100', 'imagenet','tiny-imagenet','fashion'])
parser.add_argument('--model', default='vgg16', type=str, help='Model name',
                    choices=['vgg16_imagenet','cifarnet','small','vgg16', 'resnet18', 'resnet20','alexnet','vgg16_no_bn',
                             'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed','resnet19',
                             'resnet18', 'resnet20', 'resnet34','resnet34_imagenet','resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--checkpoint', default='./saved_models', type=str, help='Directory for saving models')

parser.add_argument('--naive', default='', type=str)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epochs', default=50, type=int)

args = parser.parse_args()
args.mid = f'{args.dataset}_{args.model}'
savename = os.path.join(args.checkpoint, args.mid)+"_new"#+"_new"#+"no_bn"#+"no_bias"+"no_affine"#+"_no_affine"#+"_no_affine"+"_bn"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define Hyper-parameters 

num_epochs = 300
batch_size =args.batchsize
learning_rate = 0.001



# to extract activation values
sample = 0
n_steps = args.t
model = modelpool(args.model, args.dataset)

criterion = nn.CrossEntropyLoss()


train_loader, test_loader = datapool(args.dataset, batch_size,2,shuffle=True)
   

model.load_state_dict(torch.load(savename + '.pth'))

model.to(device)
num_relu = str(model).count('ReLU')
naive = args.naive#'_naive'#_naive

thresholds = torch.zeros(num_relu,2*n_steps)


sample = 0
print('%s_threshold_all_noaug%s%d.npy'%(savename,naive,n_steps))
def isActivation(name):
    if 'relu' in name.lower():
        return True
    return False
counter = 0



def replace_activation_by_spike(model):
    global counter
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_activation_by_spike(module)
        if isActivation(module.__class__.__name__.lower()):
            #*(1.0-(counter)*1.0/thresholds.shape[0]))
            thresholds[counter,n_steps:] = thresholds1[counter,1]/n_steps#thresholds_out_sum/n_steps# thresholds1[counter,1]/n_steps
            thresholds[counter,:n_steps] = thresholds1[counter,0]/n_steps#thresholds_inner_sum/n_steps#thresholds1[counter,0]/n_steps
            model._modules[name] = SPIKE_layer(thresholds[counter,n_steps:],thresholds[counter,0:n_steps])
            
            counter += 1
    return model

thresholds1 = torch.Tensor(np.load('%s_threshold_all_noaug%s%d.npy'%(savename,naive,1)))
print("Loaded thresholds shape:", thresholds1.shape)

model = replace_activation_by_spike(model)
model = replace_maxpool2d_by_avgpool2d(model)
model = replace_layer_by_tdlayer(model)
model.to(device)      

if n_steps>1:
    model.load_state_dict(torch.load(savename + '%s_updated_snn1_%d.pth'%(naive,n_steps-1)))
    #model = replace_activation_by_spike(model)

print("model loaded...")


train_loader, test_loader = datapool(args.dataset, batch_size,2,shuffle=True)
train_loss = []
test_loss = []
train_acc= []
test_acc = []
def test_snn(model):
    model.eval()
    with torch.no_grad():
        print(n_steps)
        correct = 0
        total = 0
        loss = 0
        
        for images, labels in test_loader:
            images = add_dimension(images,n_steps)
            images = images.to(device)
        
            labels = labels.to(device)
            
            outputs = 0
            #for t in range(n_steps):
            outputs = model(images,thresholds,L=0,t=n_steps) #N,classes,T
            outputs = torch.sum(outputs,1)
            #outputs = outputs[:,-1,...]

            
            _, predicted = torch.max(outputs.data/n_steps, 1)
            loss += criterion(outputs/n_steps, labels).item()*images.shape[0]
            #print("snn predicted",predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            print('Accuracy of the snn network on the %d test images: %f, %d are correct'%(total,100 * correct / total,correct))
        test_loss.append(loss/total)
        test_acc.append(100 * correct / total)
    return 100 * correct / total
def train_snn(train_dataloader, test_dataloader, model, epochs, device, loss_fn, lr=0.1, wd=5e-4, save=None, parallel=False, rank=0):
    model.cuda(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=wd) 
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,momentum=0.9,weight_decay=wd) 
   
    para1, para2, para3 = regular_set(model)
    
    optimizer = torch.optim.SGD([
                                {'params': para1, 'weight_decay': wd}, 
                                {'params': para2, 'weight_decay': wd}, 
                                {'params': para3, 'weight_decay': wd}
                                ],
                                lr=lr, 
                                 momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,verbose=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
    #                     milestones=[100], # List of epoch indices
    #                     gamma =0.5,verbose=True)
    best_loss = 10000
    best_epoch = 0
    best_acc = 0
    start_epoch = 0
    for epoch in range(start_epoch,epochs):
        model.train()
        epoch_loss = 0
        length = 0
        total = 0
        correct = 0
        
        for img, label in train_dataloader:
            img = add_dimension(img,n_steps)
            img = img.cuda(device)
            
            labels = label.cuda(device)
            outputs = model(img,thresholds,L=0,t=n_steps) 
            if args.TET==0:
                
                
                
                
                outputs = torch.sum(outputs,1)
                #outputs = outputs[:,-1,...]
                optimizer.zero_grad()
                
                loss = loss_fn(outputs/n_steps, labels)
                #loss = loss_fn(outputs, labels)
            else:
                
                loss = 0
                optimizer.zero_grad()
                for t in range(n_steps):
                    #print(t)
                    loss = loss_fn(outputs[:,t,...], labels)
                loss = loss/n_steps
                
                
                y = torch.zeros_like(outputs).fill_(thresholds[-1,1]*n_steps)
                
                loss = 0.9*loss + 0.1*MMDLoss(outputs,y)
            #print(loss.item()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()*img.shape[0]
            length += len(label)
            if args.TET==0:
                _, predicted = torch.max(outputs.data, 1)
            else:
                outputs = torch.mean(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total%(256*8) == 0:
                print('Epoch:%d, Accuracy of the snn network on the %d train images: %f, loss:%f'%(epoch,total,100 * correct / total,epoch_loss/total))
                #exit()
        print('Epoch:%d, Accuracy of the snn network on the %d train images: %f, loss:%f'%(epoch,total,100 * correct / total,epoch_loss/total))
        scheduler.step()
        train_loss.append(epoch_loss/total)
        train_acc.append(100 * correct / total)
        if (epoch+1)%1==0:
            test_acc = test_snn(model)
        if best_acc<=test_acc:
            torch.save(model.state_dict(),  savename + '%s_updated_snn1_%d.pth'%(naive,n_steps))
            best_acc = test_acc
            best_epoch = epoch
        print('Best acc: %f, found at the epoch: %d, with loss: %f'%(best_acc,best_epoch,best_loss))
        
        
        np.save(savename+'_updated_snn1_train_loss_%d.npy'%(n_steps),train_loss)
        np.save(savename+'_updated_snn1_test_loss_%d.npy'%(n_steps),test_loss)
        np.save(savename+'_updated_snn1_train_acc_%d.npy'%(n_steps),train_acc)
        np.save(savename+'_updated_snn1_test_acc_%d.npy'%(n_steps),test_acc)
        print("saved as: ",savename+'_updated_snn1_test_acc_%d.npy'%(n_steps))
    return model
print("Initial SNN accuracy, before training.....")


test_snn(model)


model = train_snn(train_loader, test_loader, model, args.epochs, device, criterion,args.lr, args.wd, savename)
print("Final SNN accuracy, after training.......")
test_snn(model)