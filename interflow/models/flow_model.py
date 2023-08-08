import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.FlowStep import FlowStep
from models.modules.RRDBNet_arch import RRDBNet
from models.modules.flow import GaussianDiag ,Gaussian
from models.modules import thops
from models.modules.RRDBGlowNet_arch import RRDBGlowNet



class DegradationNet(nn.Module):
    def __init__(self, in_ch=320, out_ch=48, hidden_ch=64, L=3):
        super(DegradationNet, self).__init__()
        layers = []
        for i in range(L):
            if(i == 0):
                layers.append(nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
            elif(i == L-1):
                layers.append(nn.Conv2d(in_channels=hidden_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Conv2d(in_channels=hidden_ch, out_channels=hidden_ch, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        #return self.net(x) + 0.1    # Need numerical stability in FlowLoss
        x = self.net(x) + 0.5
        #x = torch.clip(x, 0.5, 1.5)
        #x = self.net(x)
        #x = torch.sigmoid(x) + 0.5
        return x

class InterpFlow(nn.Module):
    def __init__(self, scales:list,mean_init:float=0.001,std_init:float=1.0):
        super(InterpFlow, self).__init__()
        opt = {'scale': 4,
               'network_G': {'in_nc': 3, 'out_nc': 3, 'nf': 64, 'nb': 23, 'upscale': 4, 'train_RRDB': True, 'scale': 4,
                             'flow': {'K': 16, 'L': 2, 'noInitialInj': True, 'LU_decomposed': True, 
                                      'coupling': 'CondAffineSeparatedAndCond', 'additionalFlowNoAffine': 2, 
                                      'split': {'enable': True}, 'fea_up0': True, 
                                      'stackRRDB': {'blocks': [1, 8, 15, 22], 'concat': True}, 
                                      'CondAffineSeparatedAndCond': {'eps': 0.001, 'multReverse': True}}},}
        in_nc = 3
        out_nc = 3
        nf = 64
        nb = 23
        self.scale = 4
        K = 16
        self.arch = RRDBGlowNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, scale=self.scale, K=K, opt=opt, step=0)
        self.flow_upsampler = self.arch.flowUpsamplerNet
        self.rrdb = self.arch.RRDB
        self.degradation_net = DegradationNet()
        self.quant = 256

        self.classes = scales
        trainable_mean = True
        trainable_var = True
        self.C = self.flow_upsampler.C
        std_init_shift = std_init

        self.I = torch.nn.Parameter(torch.eye(self.C, requires_grad=False), requires_grad=False)

        mean_init_shift = mean_init
        num_class = len(self.classes)
        mean_interval = torch.linspace(-mean_init_shift*(num_class//2),mean_init_shift*(num_class//2),num_class)
        self.mean_shifts = [
            torch.nn.Parameter(torch.zeros(1,requires_grad=trainable_mean) + mean_interval[mean_idx], requires_grad=trainable_mean) 
            for mean_idx, _ in enumerate(range(len(self.classes)))]
        self.cov_shifts = [
            torch.nn.Parameter(torch.eye(self.C,requires_grad=trainable_var) * std_init_shift, requires_grad=trainable_var) 
            for _ in range(len(self.classes))]
        
        for idx_shift in range(len(self.mean_shifts)):
            self.register_parameter(f"mean_shift_{idx_shift}", self.mean_shifts[idx_shift])
        for idx_shift in range(len(self.cov_shifts)):
            self.register_parameter(f"cov_shift_{idx_shift}", self.cov_shifts[idx_shift])


    def forward(self, input_value, cond_image, label=None, reverse=False, test_input_size=None, lr_enc=None, add_gt_noise = True, calc_loss=True):
        if(reverse):
            if input_value is None:
                raise NotImplementedError
            lr = cond_image
            z = input_value
            if(lr_enc is None):
                lr_enc = self.arch.rrdbPreprocessing(lr)
            # logdet = 0.
            logdet = torch.zeros_like(lr[:, 0, 0, 0])
            if add_gt_noise:
                # hr and lr have same resolutions 
                pixels = thops.pixels(lr)
                logdet = logdet - float(-np.log(self.quant) * pixels)
            z, logdet = self.flow_upsampler(rrdbResults=lr_enc, z=z, logdet=logdet, reverse=True, epses=None, y_onehot=torch.zeros(input_value.shape[0]), test_input_size=test_input_size)
            
            return z, logdet
        else:
            lr = cond_image
            gt = input_value
            logdet = torch.zeros_like(gt[:, 0, 0, 0])
            pixels = thops.pixels(gt)
            if(lr_enc is None):
                lr_enc = self.arch.rrdbPreprocessing(lr)
            if add_gt_noise: # add quantization noise      
                # only support noiseQuant
                gt = gt + ((torch.rand(gt.shape, device=gt.device) - 0.5) / self.quant)
                logdet = logdet + float(-np.log(self.quant) * pixels)
            # logdet = torch.zeros_like(gt[:, 0, 0, 0])
            z, logdet = self.flow_upsampler(rrdbResults=lr_enc, gt=gt, logdet=logdet, reverse=False, epses=None, y_onehot=torch.zeros(input_value.shape[0]))
            if not calc_loss:
                return z, logdet
            else:
                return self.calculate_loss(z,logdet,pixels,label)


    def calculate_loss(self, epses, logdet, pixels, label):
        objective = logdet.clone()
    
        if isinstance(epses, (list, tuple)):
            z = epses[-1]
        else:
            z = epses
        
        if type(label) != torch.Tensor:
            label = torch.Tensor(label)
        
        assert z.shape[0] == label.shape[0], 'need one class label per datapoint'
        assert len(label.shape) == 1, 'labels must be one dimensional'

        dom = {'X' : label == 0}
        dom.update({f'Y_{c}': label == c for c in self.classes})

        cov_shifteds = list()
        for i, cov_shifted in enumerate(self.cov_shifts):
            cov_shifteds.append(torch.matmul(cov_shifted, cov_shifted.T))
        mean_shifteds = self.mean_shifts

        ll = torch.zeros(z.shape[0], device=z.get_device() if z.get_device() >= 0 else None)

        for i,(k,v) in enumerate(dom.items()):
            if k == 'X':
                ll[v] = Gaussian.logp(None, None, z[v])
            else:
                ll[v] = Gaussian.logp(mean=mean_shifteds[i-1]*torch.ones(self.C,requires_grad=True).cuda(), cov=cov_shifteds[i-1], x=z[v])
        

        # ib loss
        classes = np.tile(self.classes,reps=z.shape[0]).reshape((z.shape[0],-1)) # (n,cls)
        classes = torch.tensor(classes).cuda()
        mask_label = (classes == label.reshape(-1,1)) # (n,cls)
        ll_ib = torch.zeros(classes.shape, device=z.get_device() if z.get_device() >= 0 else None)

        for i, (k,v) in enumerate(dom.items()):
            if k == 'X':
                ll[v] = Gaussian.logp(None, None, z[v])
            else:
                ll_ib[:,i-1] = Gaussian.logp(mean=mean_shifteds[i-1]*torch.ones(self.C,requires_grad=True).cuda(), cov=cov_shifteds[i-1], x=z)
                
        ll_ib = ll_ib / float(pixels)
        ll_ib = torch.log_softmax(ll_ib, dim=1) # (n,cls)
        ll_ib = torch.masked_select(ll_ib,mask_label) # (n)

        objective = objective + ll


        nll = (-objective) / float(np.log(2.) * pixels)

        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet, ll_ib

    def get_lr(self, input_lr, heat=1., test_input_size=None):
        z_shape = (input_lr.shape[0], input_lr.shape[1]*16, input_lr.shape[2], input_lr.shape[3])
        z = torch.normal(mean=0, std=heat, size=z_shape).cuda()
        return self.forward(z, input_lr, reverse=True, test_input_size=test_input_size)[0]


    def interpolate_image(self, input_lr1, scale1, input_lr2, scale2, target_scale, cond_gt):
        scale1 = torch.tensor(scale1).double().cuda()
        scale2 = torch.tensor(scale2).double().cuda()
        target_scale = torch.tensor(target_scale).double().cuda()

        z1, _ = self.forward(input_lr1, cond_gt, calc_loss=False)
        z2, _ = self.forward(input_lr2, cond_gt, calc_loss=False)

        a = (target_scale - scale2) / (scale1 - scale2)
        target_z = a*z1 + (1-a)*z2

        return self.forward(target_z, cond_gt, reverse=True, test_input_size=input_lr1.shape[2], calc_loss=False)[0]

    def weighted_interpolate_image(self, input_lr1, input_lr2, a, cond_gt, add_noise = False):
        z1, _ = self.forward(input_lr1, cond_gt, add_gt_noise=add_noise, calc_loss=False)
        z2, _ = self.forward(input_lr2, cond_gt, add_gt_noise=add_noise, calc_loss=False)

        target_z = (1-a)*z1 + a*z2

        return self.forward(target_z, cond_gt, reverse=True, add_gt_noise=add_noise, test_input_size=input_lr1.shape[2], calc_loss=False)[0]

    def get_pdf(self,z:torch.Tensor,scales:list):
        pixels = thops.pixels(z)
        ll = list()

        cov_shifteds = list()
        for i, cov_shifted in enumerate(self.cov_shifts):
            cov_shifteds.append(torch.matmul(cov_shifted, cov_shifted.T))
        mean_shifteds = self.mean_shifts

        for scale_idx, scale in enumerate(scales):
            index = self.classes.index(scale)
            logp = Gaussian.logp(mean=torch.ones(self.C,requires_grad=True).cuda() * mean_shifteds[index], cov=cov_shifteds[index], x=z) / float(np.log(2.) * pixels)
            ll.append(logp)
        return torch.stack(ll,dim=0).reshape(-1)


    def get_shifted_info(self):
        if len(self.mean_shifts) != 0:
            return np.stack([m.detach().cpu() for m in self.mean_shifts],axis=0), \
                    np.stack([c.detach().cpu() for c in self.cov_shifts],axis=0), \
                    self.classes
        else:
            return None, None, None