"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion,affine=True,track_running_stats=True)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion,affine=True,track_running_stats=True)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x,counter=0,L=0,t=0):
        #residual = self.residual_function(x)
        out = x
        for i in range(len(self.residual_function)):
            #print(self.residual_function[i])
            if 'SPIKE_layer' in str(self.residual_function[i]):
                out =self.residual_function[i](out,t)
            else:
                out =self.residual_function[i](out)
            if 'SPIKE_layer' in str(self.residual_function[i]):
                #print("spike",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out,counter,True
            if 'ReLU' in str(self.residual_function[i]):
                #print("relu",counter,out.sum())
                
                counter += 1    
                if counter == L:
                    return out,counter,True
         
        shortcut = self.shortcut(x)
        
        if 'SPIKE_layer' in str(self.relu):
            #print("spike",counter,out.sum())
            out = self.relu(out + shortcut,t)
            counter += 1    
            if counter == L:
                return out,counter,True
        if 'ReLU' in str(self.relu):
            #print("relu",counter,out.sum())
            out = self.relu(out + shortcut)
            counter += 1    
            if counter == L:
                return out,counter,True

        return out,counter,False
        #out = self.relu(residual + shortcut)
        #return out
        #return self.relu(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion,affine=True,track_running_stats=True),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion,affine=True,track_running_stats=True)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
    
        return self.relu(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes,bias=True)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,thresholds=0,L=0,t=0):
        out = x
        counter = 0
        for i in range(len(self.conv1)):
            if 'SPIKE_layer' in str(self.conv1[i]):
                out = self.conv1[i](out,t)
            else:
                out = self.conv1[i](out)
            #print(i,out.sum())
            #if i==1:
                #return out
            if 'SPIKE_layer' in str(self.conv1[i]):
                #print("spike",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
            if 'ReLU' in str(self.conv1[i]):
                #print("relu",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
        for i in range(len(self.conv2_x)):
            out,counter,found = self.conv2_x[i](out,counter,L,t)
            if found:
                return out
        for i in range(len(self.conv3_x)):
            out,counter,found = self.conv3_x[i](out,counter,L,t)
            if found:
                return out
        for i in range(len(self.conv4_x)):
            out,counter,found = self.conv4_x[i](out,counter,L,t)
            if found:
                return out
        for i in range(len(self.conv5_x)):
            out,counter,found = self.conv5_x[i](out,counter,L,t)
            if found:
                return out
        out = self.avg_pool(out)
        
        if out.shape[-1]==1:
            out = out.view(out.size(0), -1)
        else:
            out = out.view(out.size(0), out.size(1),out.size(-1))
        
        out = self.fc(out)
        return out
        '''
            #print(i,out.sum())
            #if i==1:
                #return out
            if 'LIFSpike' in str(self.conv2_x[i]):
                print("spike",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
            if 'ReLU' in str(self.conv2_x[i]):
                print("relu",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
        for i in range(len(self.conv3_x)):
            out = self.conv3_x[i](out)
            #print(i,out.sum())
            #if i==1:
                #return out
            if 'LIFSpike' in str(self.conv3_x[i]):
                print("spike",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
            if 'ReLU' in str(self.conv3_x[i]):
                print("relu",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
        for i in range(len(self.conv4_x)):
            out = self.conv4_x[i](out)
            #print(i,out.sum())
            #if i==1:
                #return out
            if 'LIFSpike' in str(self.conv4_x[i]):
                print("spike",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
            if 'ReLU' in str(self.conv4_x[i]):
                print("relprint("relu"u",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
        #print(self.conv5_x,len(self.conv5_x[0].residual_function))
        print(self.conv5_x[0](out,counter).shape)
        print(self.conv5_x[0],'dkdfklfbl;',self.conv5_x[1],'hai')
        for i in range(len(self.conv5_x)):
            out = self.conv5_x[i](out)
            #print(i,out.sum())
            #if i==1:
                #return out
            if 'LIFSpike' in str(self.conv5_x[i]):
                print("spike",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
            if 'ReLU' in str(self.conv5_x[i]):
                print("relu",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
        
        
        out = self.avg_pool(out)
        
        if out.shape[-1]==1:
            out = out.view(out.size(0), -1)
        else:
            out = out.view(out.size(0), out.size(1),out.size(-1))
        
        out = self.fc(out)
        return out
        '''
class ResNet4Cifar(nn.Module):
    def __init__(self, block, num_block, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,thresholds=0,L=0,t=0):
        out = x
        counter = 0
        for i in range(len(self.conv1)):
            if 'SPIKE_layer' in str(self.conv1[i]):
                out = self.conv1[i](out,t)
            else:
                out = self.conv1[i](out)
            #print(i,out.sum())
            #if i==1:
                #return out
            if 'SPIKE_layer' in str(self.conv1[i]):
                #print("spike",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
            if 'ReLU' in str(self.conv1[i]):
                #print("relu",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
       
        for i in range(len(self.conv2_x)):
            out,counter,found = self.conv2_x[i](out,counter,L,t)
            if found:
                return out
        for i in range(len(self.conv3_x)):
            out,counter,found = self.conv3_x[i](out,counter,L,t)
            if found:
                return out
        for i in range(len(self.conv4_x)):
            out,counter,found = self.conv4_x[i](out,counter,L,t)
            if found:
                return out
        output = self.avg_pool(out)
        
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def resnet18(num_classes=10, **kargs):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    
def resnet20(num_classes=10, **kargs):
    """ return a ResNet 20 object
    """
    return ResNet4Cifar(BasicBlock, [3, 3, 3], num_classes=num_classes)

def resnet34(num_classes=10, **kargs):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=10, **kargs):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=10, **kargs):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=num_classes)

def resnet152(num_classes=10, **kargs):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3],num_classes=num_classes)