
import torch
import torch.nn as nn
import time

aa = 2
sigmoid = nn.Sigmoid()
# spike layer, requires nn.Conv2d (nn.Linear) and thresh

class SpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, spike_index_inner, spike_index_outer, const):
        # 确保阈值是标量
        if isinstance(spike_index_inner, torch.Tensor):
            spike_index_inner = spike_index_inner.item()
        if isinstance(spike_index_outer, torch.Tensor):
            spike_index_outer = spike_index_outer.item()
        
        # 将标量转换为tensor以便保存
        spike_index_inner_tensor = torch.tensor(spike_index_inner)
        const_tensor = torch.tensor(const)
        
        ctx.save_for_backward(mem.clone(), 
                            const_tensor * spike_index_inner_tensor,
                            spike_index_inner_tensor)
                            
        spike = mem.ge(spike_index_inner*const).float() * spike_index_outer*const
        return spike

    @staticmethod
    def backward(ctx, grad_output):
        mem, spike_index_inner, spike_index_inner1 = ctx.saved_tensors
        
        # 确保使用标量值进行计算
        inner_val = spike_index_inner.item()
        inner_val1 = spike_index_inner1.item()
        
        # 计算surrogate gradient
        hu = abs(mem - inner_val) < inner_val1
        
        return grad_output*hu, None, None, None  # 这里改为grad_output

class SPIKE_layer(nn.Module):
    def __init__(self, thresh_inner, thresh_outer):
        super(SPIKE_layer, self).__init__()
        
        # 确保阈值是一维张量
        if isinstance(thresh_inner, (int, float)):
            thresh_inner = torch.tensor([thresh_inner])
        if isinstance(thresh_outer, (int, float)):
            thresh_outer = torch.tensor([thresh_outer])
        
        self.thresh_outer = thresh_outer.view(-1)
        self.thresh_inner = thresh_inner.view(-1)
    
    def forward(self, input, t):
        x = input
        mem = 0
        spike_pot = []
        T = x.shape[1]
        const = 1
        
        for t in range(T):
            mem += x[:, t, ...]
            # 确保索引在范围内
            t_idx = min(t, len(self.thresh_inner)-1)
            
            # 转换阈值为标量
            thresh_inner = self.thresh_inner[t_idx].item()
            thresh_outer = self.thresh_outer[t_idx].item()
            
            spike = SpikeAct.apply(mem, thresh_inner, thresh_outer, T)
            
            # soft-reset
            mem -= const * mem.ge(thresh_inner*T).float() * thresh_outer*T
            spike_pot.append(spike)
        
        return torch.stack(spike_pot, dim=1)