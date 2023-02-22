import argparse
import pickle
import time
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Baby-1000', help='Baby')
parser.add_argument('--similar_rate', type=int, default=0.3)   #两条序列的相似度阈值
opt = parser.parse_args()


dataset = opt.dataset
f = open('data/%s.txt' % dataset, 'r')      #数据集读入
User = defaultdict(list)
usernum = 0
itemnum = 0
user_train = {}     #储存user 训练序列
target = {}         #存放训练时t时刻的target
for line in f:
    u, i = line.rstrip().split(' ')
    u = int(u)
    i = int(i)
    usernum = max(u, usernum)
    itemnum = max(i, itemnum)
    User[u].append(i)
for user in User:
    nfeedback = len(User[user])
    if nfeedback < 3:
        user_train[user] = User[user]
    else:
        user_train[user] = User[user][:-2]
    target[user] = User[user][-1]       #取训练序列的最后一位
neighbor_dict = defaultdict(list)       #邻居列表
sim = defaultdict(list)                 #相似user列表，即为在对比学习中的强制正样本
source_edge = []                        #起始点
target_edge = []                        #中止点
weight = []                             #边的权重
count = 0

for u_i in tqdm(user_train):              #遍历每一个user
    for u_j in range(u_i, usernum + 1):
        if u_i == u_j:          #如果为相同user跳过
            continue
        common_items = np.intersect1d(user_train[u_i], user_train[u_j])     #得到两个序列的交集
        if len(common_items)/len(user_train[u_i]) >= opt.similar_rate:          #大于阈值为邻居节点
            source_edge.append(u_i)
            target_edge.append(u_j)
            weight.append(len(common_items)/len(user_train[u_i]))           #相似度为边的权重
            count += 1
            sim[u_i].append(u_j)                                          #相似user 作为对比学习中的正样
        # 建立双向连接
        if len(common_items)/len(user_train[u_j]) >= opt.similar_rate:          #大于阈值为邻居节点
            source_edge.append(u_j)
            target_edge.append(u_j)
            weight.append(len(common_items)/len(user_train[u_j]))               #相似度为边的权重
            count += 1
            sim[u_j].append(u_i)                                          #相似user 作为对比学习中的正样
print('ave user edge:', count / len(user_train))                            #每个user平均有多少边
rel_matrix = sp.coo_matrix((weight, (source_edge, target_edge)),shape=(usernum + 1, usernum + 1))       #生成稀疏邻接矩阵
pickle.dump([rel_matrix, sim], open('data/' + dataset + '_adj' + '.pkl', 'wb'))          #保存文件，user邻接矩阵，每个user的负采样，每个user在对比学习中的正样本user
