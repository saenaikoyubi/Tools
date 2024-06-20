# import public modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv1x1(nn.Module):
    def __init__(self, d_in_channels, d_out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels=d_in_channels, out_channels=d_out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        # Conv2dの入力/出力は(N,C,H,W)
        # N バッチサイズ
        # C チャネルの数
        # H 二次元データの縦サイズ　
        # W 二次元データの横サイズ
        
    def forward(self, x):
        x = self.conv1x1(x)
        return x
        
class GatedAxialAttentionWidth(nn.Module):
    def __init__(self, C, H, W, device='cpu'):
        super().__init__()
        
        self.Channel = C
        self.Height = H
        self.Weight = W
        self.device = device
        self.d = np.sqrt(self.Channel)
        
        self.q_conv1x1 = Conv1x1(d_in_channels=self.Channel, d_out_channels=self.Channel)
        self.k_conv1x1 = Conv1x1(d_in_channels=self.Channel, d_out_channels=self.Channel)
        self.v_conv1x1 = Conv1x1(d_in_channels=self.Channel, d_out_channels=self.Channel)
        
        self.rq = nn.Parameter(torch.empty(self.Channel, self.Weight, self.Weight).uniform_(0.1, 0.9))
        self.rk = nn.Parameter(torch.empty(self.Channel, self.Weight, self.Weight).uniform_(0.1, 0.9))
        self.rv = nn.Parameter(torch.empty(self.Channel, self.Weight, self.Weight).uniform_(0.1, 0.9))
        
        self.Gq = nn.Parameter(torch.empty(1).uniform_(0.1, 0.9))
        self.Gk = nn.Parameter(torch.empty(1).uniform_(0.1, 0.9))
        self.Gv1 = nn.Parameter(torch.empty(1).uniform_(0.1, 0.9))
        self.Gv2 = nn.Parameter(torch.empty(1).uniform_(0.1, 0.9))
        
        # self.q_bn = nn.BatchNorm2d(self.Channel)
        # self.k_bn = nn.BatchNorm2d(self.Channel)
        
    def forward(self, x:torch.tensor):
        # xのサイズは(N,C,H,W)
        # N バッチサイズ
        # C チャネルの数
        # H 二次元データの縦サイズ　
        # W 二次元データの横サイズ
        q = self.q_conv1x1(x)
        k = self.k_conv1x1(x)
        v = self.v_conv1x1(x)
        # q = self.q_bn(q)
        # k = self.k_bn(k)
        qk = torch.einsum('ncij, nciw->nijw', q, k) / self.d # width方向にchannelで内積をとる。queryと　keyの積をとって関連度のようなものを計算する。
        qrq = torch.einsum('ncij, cjw->nijw', q, self.rq) / self.d
        krk = torch.einsum('ncij, cjw->nijw', k, self.rk) / self.d
        weight = F.softmax(qk+self.Gq*qrq+self.Gk*krk, dim=3)
        x = torch.einsum('nijw, nciw->ncij', weight, self.Gv1*v) + torch.einsum('nijw, cjw->ncij', weight, self.Gv2*self.rv)
        return x

class GatedAxialAttentionHeight(nn.Module):
    def __init__(self, C, H, W, device='cpu'):
        super().__init__()

        self.Channel = C
        self.Height = H
        self.Weight = W
        self.device = device
        self.d = np.sqrt(self.Channel)
        
        self.q_conv1x1 = Conv1x1(d_in_channels=self.Channel, d_out_channels=self.Channel)
        self.k_conv1x1 = Conv1x1(d_in_channels=self.Channel, d_out_channels=self.Channel)
        self.v_conv1x1 = Conv1x1(d_in_channels=self.Channel, d_out_channels=self.Channel)
        
        self.rq = nn.Parameter(torch.empty(self.Channel, self.Height, self.Height).uniform_(0.1, 0.9))
        self.rk = nn.Parameter(torch.empty(self.Channel, self.Height, self.Height).uniform_(0.1, 0.9))
        self.rv = nn.Parameter(torch.empty(self.Channel, self.Height, self.Height).uniform_(0.1, 0.9))
        
        self.Gq = nn.Parameter(torch.empty(1).uniform_(0.1, 0.9))
        self.Gk = nn.Parameter(torch.empty(1).uniform_(0.1, 0.9))
        self.Gv1 = nn.Parameter(torch.empty(1).uniform_(0.1, 0.9))
        self.Gv2 = nn.Parameter(torch.empty(1).uniform_(0.1, 0.9))
        
        # self.q_bn = nn.BatchNorm2d(self.Channel)
        # self.k_bn = nn.BatchNorm2d(self.Channel)
        
    def forward(self, x:torch.tensor):
        # xのサイズは(N,C,H,W)
        # N バッチサイズ
        # C チャネルの数
        # H 二次元データの縦サイズ　
        # W 二次元データの横サイズ
        
        q = self.q_conv1x1(x)
        k = self.k_conv1x1(x)
        v = self.v_conv1x1(x)
        # q = self.q_bn(q)
        # k = self.k_bn(k)
        qk = torch.einsum('ncij, nchj->nihj', q, k)  / self.d# width方向にchannelで内積をとる。queryと　keyの積をとって関連度のようなものを計算する。
        qrq = torch.einsum('ncij, chi->nihj', q, self.rq) / self.d
        krk = torch.einsum('ncij, chi->nihj', k, self.rk) / self.d
        weight = F.softmax(qk+self.Gq*qrq+self.Gk*krk, dim=3)
        x = torch.einsum('nihj, nchj->ncij', weight, self.Gv1*v) + torch.einsum('nihj, chi->ncij', weight, self.Gv2*self.rv)
        return x

class MLP(nn.Module):
    def __init__(self, num, gamma=1.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features=num, out_features=int(num/gamma))
        self.fc2 = nn.Linear(in_features=int(num/gamma), out_features=num)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class ChanelAttentionModule(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_mlp = MLP(channel)
        self.max_mlp = MLP(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x).squeeze()
        max_ = self.max_pool(x).squeeze()
        avg = self.avg_mlp(avg).unsqueeze(-1).unsqueeze(-1)
        max_ = self.max_mlp(max_).unsqueeze(-1).unsqueeze(-1)
        y = self.sigmoid(avg+max_)
        return x*y

class GatedAxialAttentionBlock(nn.Module):
    def __init__(self, C, height, width):
        super().__init__()
        # self.conv1 = Conv1x1(C, C//2)
        self.atten_w1 = GatedAxialAttentionWidth(C, height, width)
        self.atten_h1 = GatedAxialAttentionHeight(C, height, width)
        # self.atten_w2 = GatedAxialAttentionWidth(C//2, height, width)
        # self.atten_h2 = GatedAxialAttentionHeight(C//2, height, width)
        # self.conv2 = Conv1x1(C//2, C)

    def forward(self, x):
        # x = self.conv1(x)
        x = self.atten_w1(x)
        x = self.atten_h1(x)
        # y = self.atten_w2(y)
        # y = self.atten_h2(y)
        # x = self.conv2(x)
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self, C, height, width):
        super().__init__()
        self.spatial_attention = GatedAxialAttentionBlock(C, height, width)

    def forward(self, x):
        x = self.spatial_attention(x)
        return x
    
class CS_Attention(nn.Module):
    def __init__(self, C, height, width):
        super().__init__()
        self.channel_attention = ChanelAttentionModule(C)
        self.spatial_attention = SpatialAttentionModule(C, height, width)

    def forward(self, x):
        y = self.channel_attention(x)
        y = self.spatial_attention(y)
    
        return x+y
    

