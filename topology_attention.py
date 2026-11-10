import torch
import torch.nn.functional as F


def AttentionMechanism(cf:torch.Tensor, hs:list[torch.Tensor]):
    """
    基于拓扑查询的注意力机制
    Args:
        cf: 融合一致特征 [B,D]
        hs: 各视图一致特征列表 每个视图[B,D]

    Returns:
        cf_attn：经过平均注意力权重加权的融合特征 [B,D]
    """

    # 获取特征维度和视图数量
    b_dim = cf.size(0)
    view_count = len(hs)

    # 融合特征的拓扑结构作为查询，融合特征作为值，每个视图一致特征的拓扑结构作为键
    V = cf
    Q_feat = F.normalize(cf,dim=1)
    #用余弦相似度作为样本的拓扑结构
    cf_topo = torch.matmul(Q_feat, Q_feat.T)
    hs_topo = []
    for v in range(view_count):
        K_feat = F.normalize(hs[v], dim=1)
        hs_topo.append(torch.matmul(K_feat, K_feat.T))
    scale_topo = torch.sqrt(torch.tensor(b_dim, dtype=torch.float32, device=cf.device))
    attention_weights = []
    for v in range(view_count):
        scores = torch.matmul(cf_topo, hs_topo[v].T) /scale_topo
        # scores = torch.matmul(cf_topo, hs_topo[v].T)
        attention_weights.append(F.softmax(scores, dim=1))
    attention_weights = torch.stack(attention_weights,dim=0)
    attention_avg = attention_weights.mean(dim=0)
    cf_attn = torch.matmul(attention_avg,V)
    return cf_attn

