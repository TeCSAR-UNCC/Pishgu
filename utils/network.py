#network.py
import torch
from torch_geometric.nn import GATConv
from utils.gin_conv2 import GINConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from types import SimpleNamespace


# CBAM stuff
################################################################################################################################

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) 
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

################################################################################################################################

class GINN(torch.nn.Module):
    def __init__(self, input_ch):
        super(GINN, self).__init__()
        
        # input_ch = 64
        output_ch = int(input_ch/2)
        self.l1 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l1.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        input_ch = output_ch
        output_ch = int(input_ch/2)
        self.l2 = torch.nn.Linear(input_ch, output_ch)
        torch.nn.init.xavier_uniform_(self.l2.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        return x

class NetGINConv(torch.nn.Module):
    def __init__(self, num_features, output_size):
        super(NetGINConv, self).__init__()
        self.num_cords = 2 # get from yaml
        self.input_steps = int(num_features/self.num_cords)

        input_ch = self.num_cords
        output_ch = 64
        self.conv2Da = torch.nn.Conv2d(input_ch, output_ch, (2, 2),stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Da.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam1 = CBAM (output_ch)
        
        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Db = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Db.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam2 = CBAM (output_ch)

        input_ch = output_ch
        output_ch = output_ch*2
        self.conv2Dc = torch.nn.Conv2d(input_ch, output_ch, (2, 1), stride=2)
        torch.nn.init.xavier_uniform_(self.conv2Dc.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.cbam3 = CBAM (output_ch)
        feature_expansion = 2
        self.fc = torch.nn.Linear(int(num_features*2),int(num_features*2*feature_expansion))
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        nn = GINN (self.input_steps*self.num_cords*2*feature_expansion)
        nn2 = GINN(self.input_steps*self.num_cords*2*feature_expansion)
        self.conv1 = GINConv(nn, nn2, train_eps=True)

        input_ch = output_ch
        output_ch = output_size
        self.conv2Dd = torch.nn.Conv2d(input_ch, output_ch, (1, 1))
        torch.nn.init.xavier_uniform_(self.conv2Dd.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        
    def forward(self, x, x_real, edge_index):
        x1 = F.leaky_relu(self.fc(x_real))
        x1 = F.leaky_relu(self.conv1(x1, edge_index))
        x1 = x1.reshape(x.shape)
        x = torch.cat((x,x1),1)
        x = F.leaky_relu(self.conv2Da(x))
        x = self.cbam1(x)
        x = F.leaky_relu(self.conv2Db(x))
        x = self.cbam2(x)
        x = F.leaky_relu(self.conv2Dc(x))
        x = self.cbam3(x)
        #Prediction
        x = F.leaky_relu(self.conv2Dd(x))
        return x