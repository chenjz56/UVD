import argparse
import os
import pickle
import time

import torch

from model import UVD
from utils import *


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Baby-1000', type=str)
parser.add_argument('--encoder', default='SASRec', type=str, help='SASRec, GRU4Rec')
parser.add_argument('--train_dir', default='train_results', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=64, type=int) 
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=400, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)          #session encoder dropout
parser.add_argument('--l2_emb', default=0.00001, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--Agnn', type=str, default='GCN', help='sgc, GCN, GAT, GraphSAGE')      #增强图卷积的方式
parser.add_argument('--Cgnn', type=str, default='sgc', help='sgc, GCN, GAT, GraphSAGE')      #图卷积方式
parser.add_argument('--num_layer', type=int, default=3)
parser.add_argument('--aug_layer_agg', type=str, default='none', help='none, sum, avg')         #增强
parser.add_argument('--con_layer_agg', type=str, default='none', help='none, sum, avg, concat') #图卷积
parser.add_argument('--node_dropout_rate', type=float, default=0.3)         #图dropout率
parser.add_argument('--nu', type=float, default=0.0001)      #对比loss 
parser.add_argument('--xi', type=float, default=0.3)          #增强loss
parser.add_argument('--sigma', type=float, default=0.4)         #用户驱动信息的比例
parser.add_argument('--user_CL', type=bool, default=True)       #是否使用对比学习
parser.add_argument('--user_AUG', type=bool, default=True)      #是否使用增强学习

args = parser.parse_args()
fargs_name = 'UVD.encoder={}.layer={}.user_CL={}.user_AUG={}.aug_gnn={}.con_gnn={}.maxlen={}.txt'
fargs_name = fargs_name.format(args.encoder, args.num_layer, args.user_CL, args.user_AUG, args.Agnn, args.Cgnn, args.maxlen)
flog_fname = 'UVD.encoder={}.layer={}.user_CL={}.user_AUG={}.aug_gnn={}.con_gnn={}.maxlen={}-log.txt'
flog_fname = flog_fname.format(args.encoder, args.num_layer, args.user_CL, args.user_AUG, args.Agnn, args.Cgnn, args.maxlen)

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)

with open(os.path.join(args.dataset + '_' + args.train_dir, fargs_name), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    

    # global dataset
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size
    train_dict_len = [len(user_train[u]) for u in user_train]
    print('max len: %d, min len:%d, avg len:%.2f' % (np.max(train_dict_len), np.min(train_dict_len), np.mean(train_dict_len)))
    plan_data = {'loss':[],'ndcg':[]}
    f = open(os.path.join(args.dataset + '_' + args.train_dir, flog_fname), 'w')

    user_adj_matrix, sem_train = pickle.load(open('data/' + args.dataset + '_adj'+ '.pkl', 'rb'))
    sampler = WarpSampler(user_train, usernum, itemnum, sem_train, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = UVD(usernum, itemnum, user_adj_matrix, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass 
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    T_idx = 0
    best_vaile = [0, 0, 0, 0, 0, 0]
    best_test = [0, 0, 0, 0, 0, 0]

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break 
        for step in range(num_batch): 
            u, seq, sem_seq, pos = sampler.next_batch() # tuples to ndarray
            u, seq, sem_seq, pos = np.array(u), np.array(seq), np.array(sem_seq), np.array(pos)
            loss = model(u, seq, sem_seq, pos)      
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()

            adam_optimizer.step()
            if step == num_batch-1:
                print("loss in epoch {} with {} iterations: {}".format(epoch, step+1, loss.item()))
        plan_data['loss'].append(loss.item())
        if epoch % 20 == 0:
            model.eval()
            t0 = time.time()
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t1 = time.time() - t0
            print('\nepoch:%d, time: %f(s), \ntest (NDCG@10: %.4f, HR@10: %.4f, MRR@10: %.4f, \n\tNDCG@20: %.4f, HR@20: %.4f, MRR@20: %.4f)'
                  % (epoch, t1, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))
            T += t1
            T_idx += 1
            best_test = record(t_test, best_test)
            f.write(str(epoch)+':'+ str(t_test) + '\n')
            f.flush()
            plan_data['ndcg'].append(t_test[0])
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'UVD.encoder={}.layer={}.user_CL={}.user_AUG={}.aug_gnn={}.con_gnn={}.maxlen={}.pth'
            fname = fname.format(args.encoder, args.num_layer, args.user_CL, args.user_AUG, args.Agnn, args.Cgnn, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print('avg test time: ', T/T_idx)
    print('best test:', best_test)
    print("Done")

