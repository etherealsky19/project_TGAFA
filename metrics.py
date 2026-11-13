from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score,accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.nn.functional import normalize
from topology_attention import AttentionMechanism


def cluster_acc(y_true, y_pred):
    """
    计算聚类准确率 ACC (使用匈牙利算法对齐标签)
    Args:
        y_true:真实标签
        y_pred:预测标签

    Returns:
        ACC值
    """
    # 将真实标签转换为整型，确保类型一致
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size, "预测标签和真实标签大小不匹配"

    # 初始化混淆矩阵
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    # 填充混淆矩阵
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # 使用匈牙利算法进行最佳类别匹配
    row, col = linear_sum_assignment(w.max() - w)
    return w[row, col].sum() / y_pred.size

# 计算纯度（Purity）
def purity(y_true, y_pred):
    # 初始化投票标签数组
    y_voted_labels = np.zeros_like(y_true)

    # 对真实标签重新编码（确保从 0 开始）
    labels = np.unique(y_true)
    label_map = {label: idx for idx, label in enumerate(labels)}
    y_true = np.vectorize(label_map.get)(y_true)

    # 为每个聚类计算投票结果
    bins = np.arange(len(labels) + 1)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        y_voted_labels[y_pred == cluster] = hist.argmax()

    # 计算并返回纯度
    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    """
    计算聚类的各种指标
    Args:
        label:真实标签
        pred:预测标签

    Returns:
        nmi、ari、acc三个指标的值
    """
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label,pred)
    return nmi,ari,acc,pur



@torch.no_grad()
def valid(model, device, dataset, batch_size, stage):
    #nums = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    features, labels = [], []
    for xs, y, _ in loader:
        xs = {v: xs[v].to(device) for v in range(dataset.num_views)}
        _, hs, cf, _, ss = model(xs)
        cf_attn = AttentionMechanism(cf,hs)
        feature = cf_attn
        if stage == 2:
            feature = torch.cat([normalize(cf_attn,dim=1)] + [normalize(s, dim=1) for s in ss], dim=1)
            #feature = torch.cat([normalize(s, dim=1) for s in ss], dim=1)
        features.append(feature.cpu())
        labels.append(y.reshape(-1).cpu())

    fea = torch.cat(features).numpy()
    label = torch.cat(labels).numpy()
    pred = KMeans(
        n_clusters=dataset.num_classes, n_init=100
    ).fit_predict(fea)
    return evaluate(label, pred)


