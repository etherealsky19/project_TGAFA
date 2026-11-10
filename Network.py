import torch.nn as nn
from torch.nn.functional import normalize
import torch

from topology_attention import AttentionMechanism


# from topology_attention import AttentionMechanism
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(500, 500),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(500, 2000),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(2000, 500),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(500, 500),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class consistency_layer(nn.Module):
    def __init__(self,feature_dim,high_feature_dim):
        super(consistency_layer, self).__init__()
        self.consistency_layer = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )

    def forward(self, x):
        return self.consistency_layer(x)

class specificity_layer(nn.Module):
    def __init__(self,feature_dim,high_feature_dim):
        super(specificity_layer, self).__init__()
        self.specificity_layer = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )

    def forward(self, x):
        return self.specificity_layer(x)

class Network(nn.Module):
    def __init__(self, view:int, dims:list[int], feature_dim:int, high_feature_dim:int, device):
        super(Network, self).__init__()
        self.view = view
        """
        encoders:编码器
        decoders:解码器
        consistency_layers:视图一致性层
        specificity_layers:视图特定性层
        fusion_layer：一致特征融合层
        """
        self.encoders = []
        self.decoders = []
        self.consistency_layers = []
        #self.specificity_layers = []
        for v in range(view):
            self.encoders.append(Encoder(dims[v], feature_dim))
            self.decoders.append(Decoder(dims[v], feature_dim))
            self.consistency_layers.append(consistency_layer(feature_dim,high_feature_dim))
            #self.specificity_layers.append(specificity_layer(feature_dim, high_feature_dim))

        # self.encoders.append(Encoder(sum(dims),feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.consistency_layers = nn.ModuleList(self.consistency_layers)
        #self.specificity_layers = nn.ModuleList(self.specificity_layers)
        self.specificity_layers = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(high_feature_dim*view, high_feature_dim),
        )
        # self.rc_layer = nn.Sequential(
        #     nn.Linear(high_feature_dim*2,feature_dim)
        # )
        self.attention_mechanism = AttentionMechanism
    def forward(self, xs:dict[int,torch.Tensor],attn=False):
        """
        前向传播计算各种特征
        Args:
            attn: 是否开启注意力
            xs: 原始特征字典

        Returns:
            zs: 低级特征列表
            hs: 高级特征（一致特征）列表
            cf: 融合特征
            xrs: 重构特征列表
            ss：特定特征列表
        """
        zs = []
        hs = []
        xrs = []
        ss = []

        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.consistency_layers[v](z)
            s = self.specificity_layers(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
            ss.append(s)
        hc = torch.cat(hs,dim=1)
        cf = self.fusion_layer(hc)
        if attn:
            cf = self.attention_mechanism(cf,hs)
        return zs,hs,cf,xrs,ss
