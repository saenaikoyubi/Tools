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


class GatedAxialAttentionWidthMultiHead(nn.Module):
    def __init__(self, C, H, W, num_heads, device='cpu'):
        super().__init__()

        assert C % num_heads == 0, "チャネル数Cはnum_headsで割り切れる必要があります。"
        
        self.Channel = C
        self.num_heads = num_heads
        self.head_dim = C // num_heads
        self.Height = H
        self.Weight = W
        self.device = device
        self.d = np.sqrt(self.Channel)
        
        self.q_conv1x1 = Conv1x1(d_in_channels=self.Channel, d_out_channels=self.Channel)
        self.k_conv1x1 = Conv1x1(d_in_channels=self.Channel, d_out_channels=self.Channel)
        self.v_conv1x1 = Conv1x1(d_in_channels=self.Channel, d_out_channels=self.Channel)
        
        self.rq = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.Weight, self.Weight).uniform_(0.1, 0.9))
        self.rk = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.Weight, self.Weight).uniform_(0.1, 0.9))
        self.rv = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.Weight, self.Weight).uniform_(0.1, 0.9))
        
        self.Gq = nn.Parameter(torch.empty(self.num_heads, 1, 1, 1).uniform_(0.1, 0.9))
        self.Gk = nn.Parameter(torch.empty(self.num_heads, 1, 1, 1).uniform_(0.1, 0.9))
        self.Gv1 = nn.Parameter(torch.empty(self.num_heads, 1, 1, 1).uniform_(0.1, 0.9))
        self.Gv2 = nn.Parameter(torch.empty(self.num_heads, 1, 1, 1).uniform_(0.1, 0.9))

        self.out_conv1x1 = Conv1x1(d_in_channels=self.Channel, d_out_channels=self.Channel)
        
        # self.q_bn = nn.BatchNorm2d(self.Channel)
        # self.k_bn = nn.BatchNorm2d(self.Channel)
        
    def forward(self, x:torch.tensor):
        # xのサイズは(N,C,H,W)
        # N バッチサイズ
        # C チャネルの数
        # H 二次元データの縦サイズ　
        # W 二次元データの横サイズ

        N, C, H, W = x.shape
        assert C == self.Channel and H == self.Height and W == self.Width, "入力テンソルのサイズが不一致です。"

        q = self.q_conv1x1(x)  # (N, C, H, W)
        k = self.k_conv1x1(x)  # (N, C, H, W)
        v = self.v_conv1x1(x)  # (N, C, H, W)

        # ヘッドごとにリシェイプ
        # 新しい形状: (N, num_heads, head_dim, H, W)
        q = q.view(N, self.num_heads, self.head_dim, H, W)
        k = k.view(N, self.num_heads, self.head_dim, H, W)
        v = v.view(N, self.num_heads, self.head_dim, H, W)

        # q = self.q_bn(q)
        # k = self.k_bn(k)
        # 注意の計算
        # qk: (N, num_heads, H, W, W)
        qk = torch.einsum('nhcij, nhciw->nhijw', q, k) / self.d # width方向にchannelで内積をとる。queryと　keyの積をとって関連度のようなものを計算する。
        qrq = torch.einsum('nhcij, chjw->nhijw', q, self.rq) / self.d
        krk = torch.einsum('nhcij, chjw->nhijw', k, self.rk) / self.d
        weight = F.softmax(qk+self.Gq*qrq+self.Gk*krk, dim=3)
        x = torch.einsum('nhijw, nhciw->nhcij', weight, self.Gv1*v) + torch.einsum('nhijw, chjw->nhcij', weight, self.Gv2*self.rv)
        # ヘッドを結合
        x = x.view(N, C, H, W)  # (N, C, H, W)
        
        # 出力投影
        x = self.out_conv1x1(x)  # (N, C, H, W)
        return x
    
class GatedAxialAttentionHeightMultiHead(nn.Module):
    def __init__(self, C, H, W, num_heads, device='cpu'):
        super().__init__()
        self.gatedaxialatten = GatedAxialAttentionWidthMultiHead(C=C, H=W, W=H, num_heads=num_heads, device=device)
    
    def forward(self, x:torch):
        x = x.permute(0,1,3,2)
        x = self.gatedaxialatten(x)
        x = x.permute(0,1,3,2)
        return x
    
class GatedAxialAttentionMultiHead(nn.Module):
    def __init__(self, C, H, W, num_heads, device='cpu'):
        super().__init__()
        self.atten_h = GatedAxialAttentionHeightMultiHead(C=C, H=W, W=H, num_heads=num_heads, device=device)
        self.atten_w = GatedAxialAttentionWidthMultiHead(C=C, H=W, W=H, num_heads=num_heads, device=device)

    def forward(self, x):
        y = self.atten_h(x)
        y = self.atten_w(y)
        return x+y