import os
import random
import numpy as np
os.environ["OMP_NUM_THREADS"] = "3"

from metrics import valid
import torch
import torch.nn as nn
import argparse
from Network import Network
from loss import ContrastiveLoss,TopologyLoss
from Logger import get_logger
from dataloader import MATKind
import pprint


def setup_seed(seed):   #设置随机种子，确保每次实验生成的随机数相同
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pretrain(epoch):
    total_loss = 0.
    mse = nn.MSELoss()

    for batch_idx,(xs,_,_) in enumerate(dataloader):
        for v in range(num_views):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _,_,_,xrs,_ = model(xs)
        loss_list = []
        for v in range(num_views):
            loss_list.append(mse(xrs[v],xs[v]))
        r_loss = sum(loss_list) #重构损失
        r_loss.backward()
        optimizer.step()
        total_loss +=r_loss.item()
    print(f"PretrainEpoch({epoch}),Loss:{total_loss:.4f}")


def consistency_train(epoch):
    total_loss = 0.
    mse = nn.MSELoss()
    use_attn = (epoch // args.attn_interval) % 2 == 1
    for batch_idx,(xs,_,_) in enumerate(dataloader):
        for v in range(num_views):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        # if epoch>= 150:
        #     _,hs,cf,xrs,_ = model(xs,True)
        # else:
        #     _, hs, cf, xrs,_ = model(xs)
        _, hs, cf, xrs,_ = model(xs,use_attn)


        loss_list = []
        for v in range(num_views):
            loss_list.append(con_loss(cf, hs[v]))
            loss_list.append(mse(xrs[v],xs[v]))
        rc_loss = sum(loss_list) #重构损失+对比损失
        rc_loss.backward()
        optimizer.step()
        total_loss +=rc_loss.item()
    print(f"ConsistencyEpoch({epoch}),Loss:{total_loss:.4f}")

def specificity_train(epoch):
    total_loss = 0.
    # mse = nn.MSELoss()

    for batch_idx,(xs,_,_) in enumerate(dataloader):
        for v in range(num_views):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _,_,_,_,ss= model(xs)

        loss_list = []
        for v in range(num_views):
            #loss_list.append(spe_loss(cf,ss[v],xs[v]))
            #loss_list.append(mse(xs[v],xrs[v]))
            loss_list.append(con_loss.forward_spe(ss[v],xs[v],args.batch_size//num_classes//4))
        sp_loss = sum(loss_list)
        sp_loss.backward()
        optimizer.step()
        total_loss +=sp_loss.item()
    print(f"SpecificityEpoch({epoch}),Loss:{total_loss:.4f}")

if __name__ =="__main__":

    #['ALOI-100', 'BRCA', 'Caltech5V', 'Dermatology', 'Digits-6V',
    # 'LandUse21', 'LGG', 'MSRCV1', 'NUSWIDEOBJ', 'Out-Scene', 'RGB-D', 'STL-10',
    # 'Synthetic3d', 'Wikipedia']
    # 'prokaryotic','YoutubeFace']

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default="Synthetic3d",help="数据集名称")
    parser.add_argument('--batch_size', default=256, type=int,help="批次大小")
    parser.add_argument("--learning_rate", default=0.0003,type=float,help="学习率")
    parser.add_argument("--weight_decay", default=0.,type=float,)
    parser.add_argument("--pre_epochs", default=300,type=int,help="预训练轮次")
    parser.add_argument("--con_epochs", default=300,type=int,help="一致特征训练轮次")
    parser.add_argument("--spe_epochs", default=300,type=int,help="特定特征训练轮次")
    parser.add_argument("--feature_dim", default=512,type=int,help="低级特征维度")
    parser.add_argument("--high_feature_dim", default=128,type=int,help="高级特征维度")
    parser.add_argument("--attn_interval", default=30, type=int, help="注意力交替开启轮数")
    parser.add_argument("--seed", default=2026, type=int, help="种子")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    logger = get_logger(args.dataset)
    data_path = "./data"
    dataset = MATKind(args.dataset,data_path)
    num_views = dataset.num_views  # 视图数量
    num_classes = dataset.num_classes  # 类别数
    nums = len(dataset)
    args.batch_size = -(-nums//6)
    dims = dataset.get_dims()  # 各视图特征维度
    dataloader = torch.utils.data.DataLoader(
        dataset= dataset,
        batch_size= args.batch_size,
        shuffle= True,
        drop_last=False
    )

    #设置种子
    setup_seed(args.seed)
    #部署网络
    model = Network(num_views,dims,
                    args.feature_dim,args.high_feature_dim,device)
    print(model)
    model = model.to(device)
    #设置优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    #设置损失函数
    con_loss = ContrastiveLoss(device)
    spe_loss = TopologyLoss()
    con_metric,spe_metric = {},{}

    # 重构预训练
    epoch = 0
    model.specificity_layers.requires_grad_(False)
    while epoch<args.pre_epochs:
        pretrain(epoch)
        epoch += 1

    #一致特征训练
    epoch = 0
    while epoch<args.con_epochs:
        consistency_train(epoch)
        epoch += 1
    model.eval()
    if epoch != 0:
        nmi,ari,acc,pur = valid(model, device, dataset,args.batch_size,1)
        con_metric = {"nmi":round(nmi,3),"ari":round(ari,3),"acc":float(round(acc,3)),"pur":float(round(pur,3)),}

    #特定特征训练
    epoch = 0
    model.train()
    model.encoders.requires_grad_(False)
    model.decoders.requires_grad_(False)
    model.consistency_layers.requires_grad_(False)
    model.fusion_layer.requires_grad_(False)
    model.specificity_layers.requires_grad_(True)
    while epoch < args.spe_epochs:
        specificity_train(epoch)
        epoch += 1
    model.eval()
    if epoch != 0:
        nmi, ari, acc,pur = valid(model, device, dataset,args.batch_size,2)
        spe_metric = {"nmi":round(nmi,3),"ari":round(ari,3),"acc":float(round(acc,3)),"pur":float(round(pur,3)),}

    #设定并记录需要的参数
    # epoch_metric = {"pre_epochs":args.pre_epochs,"con_epochs":args.con_epochs,"spe_epochs":args.spe_epochs,}
    # feature_dim = {"feature_dim":args.feature_dim,"high_feature_dim":args.high_feature_dim,}
    info = {
        "备注": "平分批次（6）",
        "con_metric":con_metric,
        "spe_metric":spe_metric,
        "epoch_metric":{"pre_epochs":args.pre_epochs,"con_epochs":args.con_epochs,"spe_epochs":args.spe_epochs,},
        "feature_dim":{"feature_dim":args.feature_dim,"high_feature_dim":args.high_feature_dim,},
        "args":{"batchsize":args.batch_size,"attn_interval":args.attn_interval,"seed":args.seed}
    }
    info_format = pprint.pformat(info, width=80, sort_dicts=False)
    logger.info(info_format)

    #保存模型
    state = model.state_dict()
    torch.save(state, './models/' + args.dataset + '.pth')
    print("模型已保存")
