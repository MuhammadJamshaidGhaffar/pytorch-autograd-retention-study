# Extracted from wavefrontshaping/pytorch-Complex-Layers; see provenance.json and LICENSE.upstream.
import torch
from torch import nn
from torch.nn import Module, Parameter, init
class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3))
            self.bias = Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features,2))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()    
        
    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:,:2],1.4142135623730951)
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)

class ComplexBatchNorm2D(_ComplexBatchNorm):     
    
    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 4)
        #self._check_input_dim(input)

        exponential_average_factor = 0.0
        

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        if self.training:
             
            # calculate mean of real and imaginary part
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])
            
            mean = torch.stack((mean_r,mean_i),dim=1)
            
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
            
            # works for 2d 
            input_r = input_r-mean_r[None, :, None, None]
            input_i = input_i-mean_i[None, :, None, None]          
        
            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1./n*input_r.pow(2).sum(dim=[0,2,3])+self.eps
            Cii = 1./n*input_i.pow(2).sum(dim=[0,2,3])+self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0,2,3])

            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]
            
                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]
                
                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]
                
        else:
            mean = self.running_mean
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]#+self.eps

            input_r = input_r-mean[None,:,0,None,None]
            input_i = input_i-mean[None,:,1,None,None]



        

            
        # caclualte the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st
                  
        input_r, input_i = Rrr[None,:,None,None]*input_r+Rri[None,:,None,None]*input_i, \
                           Rii[None,:,None,None]*input_i+Rri[None,:,None,None]*input_r
                           
        if self.affine:
            input_r, input_i = self.weight[None,:,0,None,None]*input_r+self.weight[None,:,2,None,None]*input_i+\
                               self.bias[None,:,0,None,None], \
                               self.weight[None,:,2,None,None]*input_r+self.weight[None,:,1,None,None]*input_i+\
                               self.bias[None,:,1,None,None]

        return input_r, input_i

def build(device):
    model = nn.ModuleList([nn.Conv2d(2, 4, 1),ComplexBatchNorm2D(4)]).to(device)
    return {'model':model,'optimizer':torch.optim.SGD(model.parameters(),lr=0.001),'device':device}
def run_loop(c, iterations):
    model = c["model"]
    for i in range(iterations):
        c['optimizer'].zero_grad(set_to_none=True)
        a = model[0](torch.randn(4,2,4,4,device=c['device']))
        b = model[0](torch.randn(4,2,4,4,device=c['device']))
        real, imag = model[1](a,b)
        loss = real.square().mean() + imag.square().mean()
        loss.backward()
        gradients = torch.cat([p.grad.flatten() for p in model.parameters()]).detach().cpu()
        c['optimizer'].step()
        yield torch.cat([real.detach().cpu().flatten(),imag.detach().cpu().flatten(),gradients])
def roots(c):
    return [c['model'][1].running_mean,c['model'][1].running_covar]
def cleanup(c):
    bn = c['model'][1]
    bn.running_mean = bn.running_mean.detach()
    bn.running_covar = bn.running_covar.detach()
