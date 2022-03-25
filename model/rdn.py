# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

PRIMITIVES = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
]
def make_model(args, parent=False):
    return RDN(args)


class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        self.Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(self.Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, temp, weight):
        length = torch.cuda.FloatTensor([len((weight>=0.1).nonzero())])
        params = torch.cuda.FloatTensor([0])
        if len(temp)>1:
            for i in range(len(temp)):
                temp[i] = temp[i]*weight[i]
                if weight[i]>0.1:
                    params += weight[i]*length
                    
            x = torch.cat(temp,dim=1)
        else:
            x = temp[0]
        out = self.conv(x)
        return out, params

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        self.C = C
        self.convs = nn.ModuleList()
        for c in range(C):
            self.convs.append(RDB_Conv(G0 + c*G, G))

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)
    def forward(self, x, weights):
        temp = [x]
        sum_p = torch.cuda.FloatTensor([0])
        for c in range(self.C):
            x, params = self.convs[c](temp, weights[c][:c+1])
            temp.append(x)
            sum_p+=params
        return self.LFF(torch.cat(temp,dim=1)) + temp[0],sum_p

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = 16
        kSize = 3
        self.scale = args.scale[0]
        # number of RDB blocks, conv layers, out channels
        self.D, self.C, G = {
            'A': (20, 6, 32),
            'B': (10, 6, 16),
        }['B']

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = self.C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])
        unf = 24
        # Up-sampling net
        self.upconv1 = nn.Conv2d(G0, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, args.n_colors, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._initialize_alphas()

    def new(self,args):
        model_new = RDN(args).cuda()
        for x, y in zip(model_new.change_arch_parameters(), self.change_arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        x0 = x
        x = self.SFENet1(x0)
        weights = F.softmax(self.growth_normal, dim=-1)
        #x0.register_hook(print)
        #print(weights.shape)
        sum_p = 0
        RDBs_out = []
        for i in range(self.D):
            x,params = self.RDBs[i](x, weights[self.C*i:self.C*(i+1)])
            RDBs_out.append(x)
            sum_p+=params
        x = self.GFF(torch.cat(RDBs_out,1))
        if self.scale == 2 or self.scale == 3:
            x = self.upconv1(F.interpolate(x, scale_factor=self.scale, mode='nearest'))
            x = self.lrelu(self.att1(x))
            x = self.lrelu(self.HRconv1(x))
        elif self.scale == 4:
            x = self.upconv1(F.interpolate(x, scale_factor=2, mode='nearest'))
            x = self.lrelu(self.att1(x))
            x = self.lrelu(self.HRconv1(x))
            x = self.upconv2(F.interpolate(x, scale_factor=2, mode='nearest'))
            x = self.lrelu(self.att2(x))
            x = self.lrelu(self.HRconv2(x))
        #self.sum_p = sum_p
        x = self.conv_last(x)

        x = x + F.interpolate(x0, scale_factor=self.scale, mode='bilinear', align_corners=False)

        return x,sum_p

    def _initialize_alphas(self):
        k = self.D*self.C
        num_ops = len(PRIMITIVES)
        self.growth_normal = torch.nn.Parameter(1e-3*torch.randn(k,num_ops).cuda().requires_grad_())
        self._arch_parameters = [
          self.growth_normal,
        ]

    def change_arch_parameters(self):
        return self._arch_parameters

