
import torch
import torch.nn as nn
import time

aa = 2
sigmoid = nn.Sigmoid()
# spike layer, requires nn.Conv2d (nn.Linear) and thresh
class SpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx,mem,spike_index_inner,spike_index_outer,const):
        
        ctx.save_for_backward(mem.clone(),const*spike_index_inner.clone(),spike_index_inner.clone())
        spike = mem.ge(spike_index_inner[0]*const).float() * spike_index_outer[0]*const
        return spike
    @staticmethod
    def backward(ctx,grad_output):
        grad_input = grad_output.clone()
        mem,spike_index_inner,spike_index_inner1 = ctx.saved_tensors

        hu = abs(mem - spike_index_inner[0])<(spike_index_inner1[0])
   
        return grad_input*hu,None,None,None,None
class SPIKE_layer(nn.Module):
    def __init__(self, thresh_inner,thresh_outer):
        super(SPIKE_layer, self).__init__()
        
        self.thresh_outer = thresh_outer
        self.thresh_inner = thresh_inner
      
    
    def forward(self, input,t):
       
        x = input
        mem = 0
        spike_pot = []
        T = x.shape[1]
   
        const = 1
        for t in range(T):       
            mem +=x[:, t, ...]
            spike = SpikeAct.apply(mem,self.thresh_inner[t],self.thresh_outer[t],T)
            # soft-rest
            mem -= const*mem.ge(self.thresh_inner[t]*T).float() * self.thresh_outer[t]*T
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)

