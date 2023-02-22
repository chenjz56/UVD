import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, sem_train, batch_size, maxlen, result_queue):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        now_len = len(user_train[user])             #目前序列长度
        new_len = np.random.randint(2, now_len + 1)     #合理范围内重新取长度
        user_seq= user_train[user][:new_len]


        seq = np.zeros([maxlen], dtype=np.int32)
        sem_seq = np.zeros([maxlen], dtype=np.int32)

        if len(sem_train[user]) > 0:                    #如果有正样本，则随机抽取一个
            sem_id = np.random.choice(sem_train[user],1)[0]
        else:
            sem_id = user                               #若没有正样本，则正样本为自己

        #相似序列生成
        sem_len = len(user_train[sem_id])
        sem_len = np.random.randint(1, sem_len + 1)
        sem_user_seq = user_train[sem_id][:sem_len]
        idx = maxlen - 1
        for i in reversed(sem_user_seq):
            sem_seq[idx] = i
            idx -= 1 
            if idx == -1: break
            
        #序列生成
        pos = user_seq[-1]
        idx = maxlen - 1
        ts = set(user_seq)
        for i in reversed(user_seq[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        return (user, seq, sem_seq, pos)


    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

def find_neighbors(user_train, usernum, similar_rate=0.3, num_neighbors=10):
    ## find global neighbors
    neighbor_dict = defaultdict(list)
    sim = defaultdict(list)
    negative_dict = defaultdict(list)
    for u_i in user_train:              #遍历每一个user
        neighbor_dict[u_i] = []
        negative_dict[u_i] = []
        sim[u_i] = []
        # while len(neighbor_dict[u_i]) < num_neighbors:
        #     u_j = np.random.randint(1, usernum + 1) # sampling neighbors
        #     if u_i == u_j: continue
        #     common_items = np.intersect1d(user_train[u_i], user_train[u_j])
        #     if len(common_items)/len(user_train[u_i]) >= similar_rate:
        #         neighbor_dict[u_i].append(u_j)
        #     elif len(common_items)/len(user_train[u_i]) <= similar_rate/5:
        #         negative_dict[u_i].append(u_j)
        for u_j in user_train:
            if u_i == u_j:
                continue
            common_items = np.intersect1d(user_train[u_i], user_train[u_j])
            if len(common_items)/len(user_train[u_i]) >= similar_rate:
                neighbor_dict[u_i].append(u_j)
                sim[u_i].append(len(common_items)/len(user_train[u_i]))
            elif len(common_items)/len(user_train[u_i]) <= similar_rate/5:
                negative_dict[u_i].append(u_j)

        while len(negative_dict[u_i]) < 1:
            u_j = np.random.randint(1, usernum + 1)  # not enough negative users
            if u_i == u_j: continue
            common_items = np.intersect1d(user_train[u_i], user_train[u_j])
            if len(common_items) / len(user_train[u_i]) <= similar_rate / 3:
                negative_dict[u_i].append(u_j)
    return neighbor_dict, negative_dict

def record(new, best):
    if new[0] > best[0]:
        best[0] = new[0]
    if new[1] > best[1]:
        best[1] = new[1]
    if new[2] > best[2]:
        best[2] = new[2]
    if new[3] > best[3]:
        best[3] = new[3]
    if new[4] > best[4]:
        best[4] = new[4]
    if new[5] > best[5]:
        best[5] = new[5]
    return best


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, sem_train, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      sem_train, 
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
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
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # @10
    NDCG_10 = 0.0
    HT_10 = 0.0
    MRR_10 = 0.0

    # @20
    NDCG_20 = 0.0
    HT_20 = 0.0
    MRR_20 = 0.0
    valid_user = 0.0

    def sample(u):
        # if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = list(range(1, itemnum + 1))
        item_idx[0], item_idx[test[u][0] - 1] = item_idx[test[u][0] - 1], item_idx[0]
        # item_idx = [test[u][0]]
        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in rated: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)

        return (u, seq, item_idx)


    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    batch_num = len(users)//args.batch_size
    if len(users) % args.batch_size != 0:
        batch_num += 1
    for i in range(batch_num):
        if i == (batch_num - 1):
            user_batch = users[i * args.batch_size:]
        else:
            user_batch = users[i * args.batch_size: (i + 1) * args.batch_size]
        one_batch = []
        for u in user_batch:
            if u in valid:
                one_batch.append(sample(u))
        u, seq, item_idx = zip(*one_batch)

        predictions = -model.predict(np.array(u), np.array(seq), np.array(item_idx))
        for pre in predictions:
            rank = pre.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 20:
                NDCG_20 += 1 / np.log2(rank + 2)
                HT_20 += 1
                MRR_20 += 1 / (rank + 1)

            if rank < 10:
                NDCG_10 += 1 / np.log2(rank + 2)
                HT_10 += 1
                MRR_10 += 1 / (rank + 1)
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    return NDCG_10 / valid_user, HT_10 / valid_user, MRR_10 / valid_user, NDCG_20 / valid_user, HT_20 / valid_user, MRR_20 / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    #@10
    NDCG_10 = 0.0
    HT_10 = 0.0
    MRR_10 = 0.0

    #@20
    NDCG_20 = 0.0
    HT_20 = 0.0
    MRR_20 = 0.0
    valid_user = 0.0
    def sample(u):
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = list(range(1, itemnum + 1))
        item_idx[0], item_idx[valid[u][0]- 1] = item_idx[valid[u][0]- 1], item_idx[0]
        # item_idx = [valid[u][0]]
        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in rated: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)
        return (u, seq, item_idx)


    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    batch_num = len(users)//args.batch_size
    if len(users) % args.batch_size != 0:
        batch_num += 1
    for i in range(batch_num):
        if i == (batch_num - 1):
            user_batch = users[i * args.batch_size:]
        else:
            user_batch = users[i * args.batch_size: (i + 1) * args.batch_size]
        one_batch = []
        for u in user_batch:
            if u in valid:
                one_batch.append(sample(u))
        u, seq, item_idx = zip(*one_batch)
        

        predictions = -model.predict(np.array(u), np.array(seq), np.array(item_idx))
        for pre in predictions:
            rank = pre.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 20:
                NDCG_20 += 1 / np.log2(rank + 2)
                HT_20 += 1
                MRR_20 += 1 / (rank + 1)

            if rank < 10:
                NDCG_10 += 1 / np.log2(rank + 2)
                HT_10 += 1
                MRR_10 += 1 / (rank + 1)
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    return NDCG_10 / valid_user, HT_10 / valid_user, MRR_10 / valid_user, NDCG_20 / valid_user, HT_20 / valid_user, MRR_20 / valid_user
