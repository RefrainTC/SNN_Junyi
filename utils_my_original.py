import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        #print(x_seq.flatten(0, 1).contiguous().shape)
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class tdLayer(nn.Module):
    def __init__(self, layer):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
       

    def forward(self, x):
        x_ = self.layer(x)
      
        return x_

class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y


def replace_layer_by_tdlayer(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_layer_by_tdlayer(module)
        if module.__class__.__name__ == 'Conv2d':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'Linear':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'BatchNorm2d':
            model._modules[name] = tdLayer(model._modules[name])
   
        if module.__class__.__name__ == 'AvgPool2d':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'Flatten':
            model._modules[name] = nn.Flatten(start_dim=-3,end_dim=-1)
        if module.__class__.__name__ == 'Dropout':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'AdaptiveAvgPool2d':
            model._modules[name] = tdLayer(model._modules[name])
            
        
        
        # if module.__class__.__name__ == 'AdaptiveAvgPool2d':
        #    model._modules[name] = tdLayer(model._modules[name])
        
        
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
    
