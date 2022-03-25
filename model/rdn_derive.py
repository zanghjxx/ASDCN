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
    def __init__(self, inChannels, growRate, index, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.idx = index
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, temp):
        x = []
        for i in self.idx:
            x.append(temp[i])
        x = torch.cat(x,dim=1)
        out = self.conv(x)
        return out

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, weights, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        self.weights = F.softmax(weights,dim=-1)
        self.C = C
        self.convs = nn.ModuleList()
        for c in range(C):
            idx = (self.weights[c][:c+1]>=0.1).nonzero()
            if len(idx)==0:
                idx = range(c)
            self.convs.append(RDB_Conv(G0 + (len(idx)-1)*G, G, idx))

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        temp = [x]
        for c in range(self.C):
            x = self.convs[c](temp)
            temp.append(x)
        return self.LFF(torch.cat(temp,dim=1)) + temp[0]

class RDN(nn.Module):
    def __init__(self, args, alpha):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = 16
        kSize = 3
        self.weights = alpha['growth_normal']

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
                RDB(growRate0 = G0, growRate = G, nConvLayers = self.C, weights = self.weights[self.C*i:self.C*(i+1)])
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
        self.scale = args.scale[0]
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, args.n_colors, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x0 = x
        x = self.SFENet1(x)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

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

        return x
