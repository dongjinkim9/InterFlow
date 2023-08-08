import torch
from torch import nn as nn

import models.modules
import models.modules.Permutations
from models.modules import flow, thops
from models.modules.FlowAffineCouplings import AffineCoupling, CondAffineCoupling, CondFtAffineCoupling, CondNormAffineCoupling
from models.modules.FlowAffineCouplingsAblation import CondAffineSeparated, CondAffineSeparatedAndCond
from models.modules.FlowActNorms import ActNorm2d, ConditionalActNormImageInjector
from utils.util import opt_get


def getConditional(rrdbResults, position):
    img_ft = rrdbResults if isinstance(rrdbResults, torch.Tensor) else rrdbResults[position]
    return img_ft


def f(in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
    layers = [flow.Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

    for _ in range(n_hidden_layers):
        layers.append(flow.Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
        layers.append(nn.ReLU(inplace=False))
    layers.append(flow.Conv2dZeros(hidden_channels, out_channels))

    return nn.Sequential(*layers)


class FlowStep(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "RandomRotation": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive",
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None):
        # check configures
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector

        self.norm_type = normOpt['type'] if normOpt else 'ActNorm2d'
        self.position = normOpt['position'] if normOpt else None

        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt

        # 1. actnorm
        if self.norm_type == "ConditionalActNormImageInjector":
            mode = normOpt.get('mode', 'scalar')
            self.actnorm = ConditionalActNormImageInjector(in_channels=64,out_channels=in_channels,
                                                           scale=1,mode=mode)
        elif self.norm_type == "noNorm":
            pass
        else:
            self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            min_singular = opt_get(opt, ["network_G", "flow", "InvertibleConv1x1", "min_singular"], None)
            self.invconv = models.modules.Permutations.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed, min_singular=min_singular)
        elif flow_permutation == "shuffle":
            self.shuffle = models.modules.Permutations.Permute2d(in_channels, shuffle=True)
        elif flow_permutation == "RandomRotation":
            self.invconv = models.modules.Permutations.RandomRotation(in_channels)
        else:
            self.reverse = models.modules.Permutations.Permute2d(in_channels, shuffle=False)

        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine" or \
                ((flow_coupling == "condAffine" or flow_coupling == "condFtAffine")
                 and acOpt is not None and acOpt.get('levels') is not None and position not in acOpt['levels']):
            kernel_hidden = opt['network_G']['flow'].get('kernel_hidden', 1)
            n_hidden_layers = opt['network_G']['flow'].get('n_hidden_layers', 1)
            self.f = f(in_channels // 2, in_channels, hidden_channels, kernel_hidden=kernel_hidden,
                       n_hidden_layers=n_hidden_layers)
            self.affine_eps = opt['network_G']['flow']['affine_coupling']['eps'] \
                if 'affine_coupling' in opt['network_G']['flow'].keys() else 0
        elif flow_coupling == "noCoupling":
            pass
        elif flow_coupling == "affineCustom":
            self.affine = AffineCoupling(acOpt=acOpt)
        elif flow_coupling == "condAffine":
            self.affine = CondAffineCoupling(acOpt=acOpt,in_channels=in_channels)
        elif flow_coupling == "condFtAffine":
            self.affine = CondFtAffineCoupling(acOpt=acOpt,in_channels=in_channels)
        elif flow_coupling == "condNormAffine":
            self.affine = CondNormAffineCoupling(acOpt=acOpt,in_channels=in_channels)
        elif flow_coupling == "CondAffineSeparated":
            self.affine = CondAffineSeparated(in_channels=in_channels)
        elif flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = CondAffineSeparatedAndCond(in_channels=in_channels,opt=opt)
        else:
            raise RuntimeError("coupling not Found")

    def forward(self, input, logdet=None, reverse=False, rrdbResults=None):
        if not reverse:
            return self.normal_flow(input, logdet, rrdbResults)
        else:
            return self.reverse_flow(input, logdet, rrdbResults)

    def normal_flow(self, z, logdet, rrdbResults=None):
        # 1. actnorm
        if self.norm_type == "ConditionalActNormImageInjector":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.actnorm(z, img_ft=img_ft, logdet=logdet, reverse=False)
        elif self.norm_type == "noNorm":
            pass
        else:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)
        
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)

        need_features = self.affine_need_features()

        # 3. coupling
        if self.flow_coupling == "additive":
            z1, z2 = thops.split_feature(z, "split")
            z2 = z2 + self.f(z1)
            z = thops.cat_feature(z1, z2)
        elif self.flow_coupling == "affine" or \
                ((self.flow_coupling == "condAffine" or self.flow_coupling == "condFtAffine")
                 and self.acOpt['levels'] is not None and self.position not in
                 self.acOpt['levels']):
            z1, z2 = thops.split_feature(z, "split")
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
            z = thops.cat_feature(z1, z2)
        elif need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=False, ft=img_ft)
        elif self.flow_coupling == "noCoupling":
            pass
        elif self.flow_coupling == "affineCustom":
            z, logdet = self.affine(z, logdet, reverse=False)
        return z, logdet

    def reverse_flow(self, z, logdet, rrdbResults=None):
        need_features = self.affine_need_features()

        # 1.coupling
        if self.flow_coupling == "additive":
            z1, z2 = thops.split_feature(z, "split")
            z2 = z2 - self.f(z1)
            z = thops.cat_feature(z1, z2)
        elif self.flow_coupling == "affine" or \
                ((self.flow_coupling == "condAffine" or self.flow_coupling == "condFtAffine")
                 and self.acOpt['levels'] is not None and self.position not in
                 self.acOpt['levels']):
            z1, z2 = thops.split_feature(z, "split")
            h = self.f(z1)
            shift, scale = thops.split_feature(h, "cross")
            scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
            z = thops.cat_feature(z1, z2)

        elif self.flow_coupling == "noCoupling":
            pass
        elif need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=True, ft=img_ft)

        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)

        # 3. actnorm
        if self.norm_type == "ConditionalActNormImageInjector":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.actnorm(z, img_ft=img_ft, logdet=logdet, reverse=True)
        elif self.norm_type == "noNorm":
            pass
        else:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features
