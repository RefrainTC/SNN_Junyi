import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        if len(x_seq.shape) == 2:  # 如果已经是2D (batch_size, features)
            b = x_seq.shape[0]
            x_seq = x_seq.view(b, 1, -1)  # 添加时间维度
            
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        x_reshaped = x_seq.reshape(-1, *x_seq.shape[2:])
        y_seq = self.module(x_reshaped)
        
        if len(y_seq.shape) == 2:  # 全连接层输出
            return y_seq.view(y_shape[0], y_shape[1], -1)
        else:  # 卷积层输出
            y_shape.extend(y_seq.shape[1:])
            return y_seq.view(y_shape)

class tdLayer(nn.Module):
    def __init__(self, layer):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
       
    def forward(self, x):
        # print(f"tdLayer input shape: {x.shape}")  # 添加这行
        x_ = self.layer(x)
        # print(f"tdLayer output shape: {x_.shape}")  # 添加这行
        return x_

def replace_layer_by_tdlayer(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_layer_by_tdlayer(module)
            
        if module.__class__.__name__ == 'Conv2d':
            model._modules[name] = tdLayer(model._modules[name])
            
        elif module.__class__.__name__ == 'Linear':
            # 直接使用tdLayer，不需要额外的Flatten
            model._modules[name] = tdLayer(module)
            
        elif module.__class__.__name__ == 'BatchNorm2d':
            model._modules[name] = tdLayer(model._modules[name])
   
        elif module.__class__.__name__ == 'AvgPool2d':
            model._modules[name] = tdLayer(model._modules[name])
            
        elif module.__class__.__name__ == 'Flatten':
            # 保持原始Flatten层的处理方式
            model._modules[name] = nn.Flatten(start_dim=-3, end_dim=-1)
            
        elif module.__class__.__name__ == 'Dropout':
            model._modules[name] = tdLayer(model._modules[name])
            
        elif module.__class__.__name__ == 'AdaptiveAvgPool2d':
            model._modules[name] = tdLayer(model._modules[name])
            
    return model

def isActivation(name):
    if 'spike_layer' in name.lower() :
        return True
    return False
def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model


def add_dimension(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x
def isActivation_spike(name):
        if 'spike_layer' in name.lower():
            return True
        return False
    
