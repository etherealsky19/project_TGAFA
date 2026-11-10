import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self,device):
        """

        Args:
            batch_size:  批大小（构造负样本掩码矩阵用）
            device: 设备
        """

        super().__init__()
        self.temperature = 1.0  # 温度参数：缩放相似度矩阵
        self.device = device

    def forward(self, cf,h):
        """
        对比学习损失
        Args:
            h: 每个视图的一致特征
            cf: 经过注意力融合的融合一致特征

        Returns:

        """
        N = h.size(0)
        h = F.normalize(h, dim=1)
        cf = F.normalize(cf, dim=1)
        similarity_matrix = torch.matmul(h, cf.T) / self.temperature
        positives = torch.diag(similarity_matrix)
        mask = torch.ones((N, N), device=self.device)
        mask.fill_diagonal_(0)
        numerator = torch.exp(positives)
        denominator = torch.sum(torch.exp(similarity_matrix) * mask, dim=1)
        loss = -torch.log(numerator / (denominator + 1e-8))
        return loss.mean()

    def forward_spe(self, s, x, k, knn_metric="euclidean"):
        """
        s: [N, D_s] 特定表示，用于对比学习
        x: [N, D_x] 原始特征，用于构建 KNN 正样本
        k: 每个样本的正样本邻居数量
        knn_metric: "euclidean" 或 "cosine"
        """
        N = s.size(0)
        if N < 2:
            return s.sum() * 0

        if x.size(0) != N:
            raise ValueError("x 和 s 的 batch 大小必须一致")

        k = min(int(k), N - 1)
        if k < 1:
            raise ValueError("k 必须大于 0")

        # 1. 根据原始特征 x 选择 KNN 正样本
        with torch.no_grad():
            x_knn = x.detach().reshape(N, -1).float()

            if knn_metric == "euclidean":
                # 欧氏距离：越小越相似
                distance = torch.cdist(x_knn, x_knn, p=2)
                distance.fill_diagonal_(float("inf"))

                indices = distance.topk(
                    k, dim=1, largest=False
                ).indices

            elif knn_metric == "cosine":
                # 余弦相似度：越大越相似
                x_knn = F.normalize(x_knn, dim=1)
                similarity_x = x_knn @ x_knn.T
                similarity_x.fill_diagonal_(float("-inf"))

                indices = similarity_x.topk(
                    k, dim=1, largest=True
                ).indices

            else:
                raise ValueError(
                    "knn_metric 必须是 'euclidean' 或 'cosine'"
                )

        # 2. 在特定表示 s 上计算对比相似度
        s = F.normalize(s, dim=1)
        similarity_s = s @ s.T

        self_mask = torch.eye(
            N, dtype=torch.bool, device=s.device
        )

        logits = similarity_s / self.temperature
        logits = logits.masked_fill(
            self_mask, float("-inf")
        )

        # 3. 多正样本对比学习
        # 分母包含所有非自身样本（包括正样本）
        log_prob = F.log_softmax(logits, dim=1)

        loss = -log_prob.gather(1, indices).mean()

        return loss


class TopologyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,cf,s,z):
        """
        拓扑一致损失
        Args:
            cf: 融合一致特征
            s: 特定特征
            z: 低级特征

        Returns:

        """
        cf = F.normalize(cf,dim=1)
        s =  F.normalize(s,dim=1)
        # cfs = cf + s
        cfs = torch.cat((cf,s),dim=1)
        z_norm = F.normalize(z,dim=1)
        z_sim = torch.softmax(torch.matmul(z_norm,z_norm.T),dim=1)
        cfs_norm = F.normalize(cfs, dim=1)
        cfs_sim = torch.softmax(torch.matmul(cfs_norm, cfs_norm.T), dim=1)
        loss = F.kl_div(torch.log(cfs_sim),z_sim,reduction='batchmean')
        return loss
