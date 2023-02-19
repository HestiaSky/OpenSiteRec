import numpy as np
import pandas as pd
import torch
import argparse
import time
from data_utils import OpenSiteRec, split
from eval_utils import PrecisionRecall_atK, NDCG_atK, get_label
from model import VanillaMF, NeuMF, RankNet, BasicCTRModel, WideDeep, DeepFM, xDeepFM, NGCF, LightGCN


MODEL = {'VanillaMF': VanillaMF, 'NeuMF': NeuMF, 'RankNet': RankNet,
         'DNN': BasicCTRModel, 'WideDeep': WideDeep, 'DeepFM': DeepFM, 'xDeepFM': xDeepFM,
         'NGCF': NGCF, 'LightGCN': LightGCN}


def parse_args():
    config_args = {
        'lr': 0.001,
        'dropout': 0.3,
        'cuda': -1,
        'epochs': 300,
        'weight_decay': 1e-4,
        'seed': 42,
        'model': 'LightGCN',
        'dim': 100,
        'city': 'Tokyo',
        'threshold': 5,
        'topk': [20],
        'patience': 5,
        'eval_freq': 10,
        'lr_reduce_freq': 10,
        'batch_size': 128,
        'save': 0,
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val)
    args = parser.parse_args()
    return args


args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

split(args.city, args.threshold)
dataset = OpenSiteRec(args)
print(dataset.testDataSize)
args.user_num, args.item_num, args.cate_num = dataset.n_user, dataset.m_item, dataset.k_cate
args.Graph = dataset.Graph
model = MODEL[args.model](args)
print(str(model))
if args.cuda is not None and int(args.cuda) >= 0:
    model = model.to(args.device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
tot_params = sum([np.prod(p.size()) for p in model.parameters()])
print(f'Total number of parameters: {tot_params}')


def train():
    model.train()
    dataset.init_batches()
    batch_num = dataset.n_user // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.n_user \
            else torch.arange(i * args.batch_size, dataset.n_user)
        users, labels = torch.LongTensor(dataset.U[indices]).to(args.device), \
                        torch.FloatTensor(dataset.bI[indices]).to(args.device)

        ratings = model(users)
        loss = model.loss_func(ratings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def train_graph():
    model.train()
    model.mode = 'train'
    dataset.uniform_sampling()
    batch_num = dataset.trainDataSize // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.trainDataSize \
            else torch.arange(i * args.batch_size, dataset.trainDataSize)
        batch = dataset.S[indices]
        users, pos_items, neg_items = torch.LongTensor(batch[:, 0]).to(args.device), \
                                      torch.LongTensor(batch[:, 1]).to(args.device), \
                                      torch.LongTensor(batch[:, 2]).to(args.device)

        loss, reg_loss = model.bpr_loss(users, pos_items, neg_items)
        loss = loss + args.weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def train_CTR():
    model.train()
    dataset.init_batches()
    batch_num = dataset.n_user // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.n_user \
            else torch.arange(i * args.batch_size, dataset.n_user)
        instances = {'Brand_ID': torch.LongTensor(dataset.U[indices]).to(args.device),
                     'Cate1_ID': torch.LongTensor(dataset.bF[indices][:, 0]).to(args.device),
                     'Cate2_ID': torch.LongTensor(dataset.bF[indices][:, 1]).to(args.device),
                     'Cate3_ID': torch.LongTensor(dataset.bF[indices][:, 2]).to(args.device)}
        labels = torch.FloatTensor(dataset.bI[indices]).to(args.device)

        ratings = model(instances)
        loss = model.loss_func(ratings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def test():
    global best_rec, best_ndcg
    model.eval()
    if args.model in ['RankNet', 'NGCF', 'LightGCN']:
        model.mode = 'test'
    testDict = dataset.testDict
    all_pos = dataset.allPos
    rec, ndcg = 0., 0.
    with torch.no_grad():
        users = list(testDict.keys())
        items = [testDict[u] for u in users]
        batch_num = len(users) // args.batch_size + 1
        for i in range(batch_num):
            batch_users = users[i * args.batch_size: (i + 1) * args.batch_size] \
                if (i + 1) * args.batch_size <= len(users) else users[i * args.batch_size:]
            # batch_pos = [all_pos[u] for u in batch_users]
            # batch_items = [[it for it in items[u] if it not in all_pos[u]] for u in batch_users]
            batch_items = [items[u] for u in batch_users]
            if args.model in ['DNN', 'WideDeep', 'DeepFM', 'xDeepFM']:
                instances = {'Brand_ID': torch.LongTensor(dataset.U[batch_users]).to(args.device),
                         'Cate1_ID': torch.LongTensor(dataset.F[batch_users][:, 0]).to(args.device),
                         'Cate2_ID': torch.LongTensor(dataset.F[batch_users][:, 1]).to(args.device),
                         'Cate3_ID': torch.LongTensor(dataset.F[batch_users][:, 2]).to(args.device)}
            else:
                instances = torch.LongTensor(batch_users).to(args.device)

            ratings = model(instances)
            ratings = ratings * dataset.lt_mask

            # exclude_index = []
            # exclude_items = []
            # for range_i, its in enumerate(batch_pos):
            #     exclude_index.extend([range_i] * len(its))
            #     exclude_items.extend(its)
            # ratings[exclude_index, exclude_items] = -(1 << 10)
            _, ratings_K = torch.topk(ratings, k=args.topk[-1])
            ratings_K = ratings_K.cpu().numpy()

            r = get_label(batch_items, ratings_K)
            for k in args.topk:
                _, batch_rec = PrecisionRecall_atK(batch_items, r, k)
                batch_ndcg = NDCG_atK(batch_items, r, k)
                rec += batch_rec * len(batch_users)
                ndcg += batch_ndcg * len(batch_users)

        rec /= len(users)
        ndcg /= len(users)
        if best_rec < rec:
            best_rec = rec
        if best_ndcg < ndcg:
            best_ndcg = ndcg
        print(f'Recall@{k}: {rec}\nnDCG@{k}: {ndcg}')


t_total = time.time()
best_rec, best_ndcg = 0., 0.
for epoch in range(args.epochs):
    if args.model in ['RankNet', 'NGCF', 'LightGCN']:
        train_graph()
    elif args.model in ['DNN', 'WideDeep', 'DeepFM', 'xDeepFM']:
        train_CTR()
    else:
        train()
    torch.cuda.empty_cache()
    if (epoch + 1) % args.eval_freq == 0:
        print(f'Epoch {epoch}')
        test()
        torch.cuda.empty_cache()

print(f'Best Results: \nRecall@{args.topk[-1]}: {round(best_rec, 4)}\nnDCG@{args.topk[-1]}: {round(best_ndcg, 4)}')

