# -*- coding: utf-8 -*-
import os
import torch
# from textwrap import fill
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
from Preprocess.augment import Cutout, CIFAR10Policy
# from Preprocess.cifar10_dvs import CIFAR10DVS
from PIL import Image, ImageEnhance, ImageOps
import random
import numpy as np
# ## Change to your own data dir
DIR = {'SVHN': './datasets','Fashion':'./datasets','CIFAR10': './datasets', 'CIFAR100': './datasets','ImageNet': './datasets', 'MNIST': './datasets'}

# code from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py 
# Improved Regularization of Convolutional Neural Networks with Cutout.

class Cutout(object):

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# code from https://github.com/yhhhli/SNN_Calibration/blob/master/data/autoaugment.py

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int32),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class CIFAR10Policy(object):

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)


def PreProcess_Cifar10(data_dir, batch_size):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(data_dir, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(data_dir, train=False, transform=trans, download=True) 

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
 
    return train_dataloader, test_dataloader


def PreProcess_Cifar100(data_dir, batch_size):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                  Cutout(n_holes=1, length=8)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    train_data = datasets.CIFAR100(data_dir, train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(data_dir, train=False, transform=trans, download=True)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader



def GetCifar10_0(batch_size, num_workers, attack=False):
    if attack:
        trans_t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16)
            ])
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans_t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(n_holes=1, length=8)
            ])
        trans = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dataloader, test_dataloader


def GetCifar10(batch_size, num_workers, shuffle=True,attack=False):
    if shuffle==True:
        trans_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    CIFAR10Policy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    Cutout(n_holes=1, length=16)
                                    ])
    else:
        
        trans_train = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                    #transforms.RandomHorizontalFlip(),
                                    #CIFAR10Policy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    #Cutout(n_holes=1, length=16)
                                    ])

    if attack:
        trans_test = transforms.Compose([transforms.ToTensor()])
    else:
        trans_test = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            #transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            ])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_train, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans_test, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, test_dataloader

def GetSVHN(batch_size, num_workers, shuffle=True,attack=False):
    if shuffle==True:
        trans_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    CIFAR10Policy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    Cutout(n_holes=1, length=16)
                                    ])
    else:
        
        trans_train = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                    #transforms.RandomHorizontalFlip(),
                                    #CIFAR10Policy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    #Cutout(n_holes=1, length=16)
                                    ])

    if attack:
        trans_test = transforms.Compose([transforms.ToTensor()])
    else:
        trans_test = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            #transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            ])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_train, download=True)
    test_data = datasets.SVHN(DIR['SVHN'], split='test', transform=trans_test, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, test_dataloader

def GetCifar_naive(data_path, dataset, batch_size, num_workers):
    """ My definition """
    if dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        data_normalization = [0.4914, 0.4822, 0.4465, 0.2023, 0.1994, 0.2010]
    elif dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        data_normalization = [0.5071, 0.4867, 0.4408, 0.2675, 0.2565, 0.2761]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((data_normalization[0], data_normalization[1], data_normalization[2]),
                             (data_normalization[3], data_normalization[4], data_normalization[5])),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((data_normalization[0], data_normalization[1], data_normalization[2]),
                             (data_normalization[3], data_normalization[4], data_normalization[5])),
    ])
    trainset = dataloader(root=data_path, train=True, download=True, transform=transform_train)
    testset = dataloader(root=data_path, train=False, download=True, transform=transform_test)
    train_loader = data.DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(testset, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def GetCIFAR10DVS(timesteps, batch_size, num_workers,):
    """ definition of 'CIFAR10DVS'
    """
    transform_train = transforms.Compose([
        transforms.Resize([48, 48]),
        transforms.RandomCrop(48, padding=4),
    ])
    trainset = CIFAR10DVS(
        DIR['CIFAR10DVS'], train=True, split_ratio=0.9, use_frame=True, 
        frames_num=timesteps, split_by='number', normalization=None, 
        transform=transform_train)
    trainloader = data.DataLoader(
        dataset=trainset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False)

    transform_test = transforms.Compose([transforms.Resize([48, 48])])

    testset = CIFAR10DVS(
        DIR['CIFAR10DVS'], train=False, split_ratio=0.9, use_frame=True,
        frames_num=timesteps, split_by='number', normalization=None, 
        transform=transform_test)
    testloader = data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False)
    return trainloader, testloader 



def GetCifar100(batch_size, num_workers, shuffle=True):
    # data_normalization = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    # tmean = [n/255. for n in [129.3, 124.1, 112.4]]
    # tstd = [n/255. for n in [68.2,  65.4,  70.4]]
    trans_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                  Cutout(n_holes=1, length=8)
                                  ])   
    trans_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])
        ])
    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_train, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans_test, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dataloader, test_dataloader


def GetImageNet(batchsize, num_workers):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    
    trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'imagenet_train'), transform=trans_t)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader =DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8,  pin_memory=True)

    test_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'imagenet_validation'), transform=trans)
    #test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2, ) 
    return train_dataloader, test_dataloader


def GetMnist(batch_size, num_workers):
    # Define a transform
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor()])  # ToTensor() from 0-255 to 0-1
    
    # transform = transforms.Compose(
    #     [transforms.Resize((28, 28)),
    #     transforms.Grayscale(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0,), (1,))])
    
    # Create Datasets and DataLoaders
    trainset = datasets.MNIST(root=DIR['MNIST'], train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=DIR['MNIST'], train=False, download=True, transform=transform)
    train_dataloader = data.DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    test_dataloader = data.DataLoader(testset, batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
    
    return train_dataloader, test_dataloader
def GetFashion(batch_size, num_workers,shuffle=True):
    # Define a transform
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor()])  # ToTensor() from 0-255 to 0-1
    
    # transform = transforms.Compose(
    #     [transforms.Resize((28, 28)),
    #     transforms.Grayscale(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0,), (1,))])
    
    # Create Datasets and DataLoaders
    trainset = datasets.FashionMNIST(root=DIR['Fashion'], train=True, download=True, transform=transform)
    testset = datasets.FashionMNIST(root=DIR['Fashion'], train=False, download=True, transform=transform)
    train_dataloader = data.DataLoader(trainset, batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    test_dataloader = data.DataLoader(testset, batch_size, shuffle=False, num_workers=num_workers,drop_last=False)
    
    return train_dataloader, test_dataloader
    
    

