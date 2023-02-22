from graph_model import *
from encoder import *
from torch_scatter import scatter_add




class UVD(torch.nn.Module):
    def __init__(self, user_num, item_num, user_adj_matrix, args):
        super(UVD, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.maxlen = args.maxlen
        self.dev = args.device
        self.user_adj_matrix = torch.sparse.FloatTensor(torch.LongTensor([user_adj_matrix.row, user_adj_matrix.col]),torch.ones(user_adj_matrix.data.shape[0]), torch.Size([self.user_num + 1,self.user_num + 1])).to(self.dev).float()

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num+1, args.hidden_units, padding_idx=0)
        self.uvd_p = None
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) 
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.user_adj_norm = self.norm_sparse_adj(self.user_adj_matrix)
        self.Agnn = args.Agnn
        self.Cgnn = args.Cgnn
        self.encoder_modle_name = args.encoder
        self.encoder = eval(args.encoder)(args)

        #GNN parameter
        self.user_aug = args.user_AUG
        if self.Agnn != 'sgc':
            self.aug_gnn = eval(self.Agnn)(args.hidden_units, args.hidden_units,layer=args.num_layer)
        if self.Cgnn != 'sgc':
            self.con_gnn = eval(self.Cgnn)(args.hidden_units, args.hidden_units,layer=args.num_layer)
        self.num_layer = args.num_layer
        self.aug_layer_agg = args.aug_layer_agg
        self.con_layer_agg = args.con_layer_agg
        if self.con_layer_agg == 'concat':
            self.con_layer_w = torch.nn.Linear(args.hidden_units * (args.num_layer + 1), args.hidden_units)
        self.nu = args.nu             #对比loss
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()       #图增强loss计算
        self.kprob = 1 - args.node_dropout_rate     #图节点留存率
        #CL
        self.user_cl = args.user_CL
        self.aug_nce_fct = torch.nn.CrossEntropyLoss()          #对比学习loss计算
        self.mask_default = self.mask_correlated_samples(batch_size=args.batch_size)
        
        self.xi = args.xi
        self.sigma = args.sigma
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def norm_sparse_adj(self, G):           #sparse adj normalized
        edge_index = G._indices()
        edge_weight = G._values()
        num_nodes= G.size(0)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        shape = G.shape
        return torch.sparse.FloatTensor(edge_index, values, shape) 

    def SparseDropout(self, g):             #sparse adj dropout
        if self.kprob == 1:         #no dropout, return immediately
            return g
        mask=((torch.rand(g._values().size())+(self.kprob)).floor()).type(torch.bool)
        rc=g._indices()[:,mask]
        val=g._values()[mask]*(1.0/self.kprob)
        G = torch.sparse.FloatTensor(rc, val,torch.Size([self.user_num + 1,self.user_num + 1]))
        return G

    def graph_augmentation(self, adj, X):
        norm_daj = adj
        item_embedding_final = [X]
        layer = X
        if self.Agnn == 'sgc':
            for i in range(self.num_layer):
                layer = torch.matmul(norm_daj, layer)
                item_embedding_final += [layer]
        else:
            layer = self.aug_gnn(layer, norm_daj)
            item_embedding_final += [layer]
        if self.aug_layer_agg == 'sum':
            item_embedding_final = torch.sum(torch.stack(item_embedding_final,dim=-1),dim=-1)
        elif self.aug_layer_agg == 'avg':
            item_embedding_final = torch.mean(torch.stack(item_embedding_final,dim=-1),dim=-1)
        else:
            item_embedding_final = item_embedding_final[-1]
        M = torch.tanh(torch.matmul(item_embedding_final, item_embedding_final.t()))
        #去除排名倒数的边
        edge = adj._indices()                                           #得到矩阵的所有边
        n_add_rem = int((1-self.kprob) * len(edge[0]))                  #得到保留和删除边的数量
        neg_probs = M[edge[0],edge[1]]                                  #取出现有的边的几率
        e_index_2b_remove = torch.argsort(neg_probs)[:n_add_rem]        #从小到大排序
        mask = torch.ones(len(edge[0]), dtype=bool, device=self.dev)
        mask[e_index_2b_remove] = False
        edge = edge[:,mask]
        val = adj._values()[mask]
        #增加排名靠前的边
        M[edge[0],edge[1]] = -1                                         #删除原来存在的边
        pos_probs = M.view(-1)
        e_index_2b_add = torch.argsort(pos_probs)[-n_add_rem:]          #取排名高的一部分
        new_val = pos_probs[e_index_2b_add]
        new_val = new_val[new_val>0]                                    #选取大于0的值
        e_index_2b_add = e_index_2b_add[new_val>0]
        new_i = e_index_2b_add // M.shape[0]                            #起始点坐标
        new_y = e_index_2b_add % M.shape[0]
        new_edge = torch.vstack((new_i,new_y))
        all_edge = torch.cat([edge,new_edge],dim=1)
        all_val = torch.cat([val,new_val],dim=0)
        #add I
        I_edge = torch.vstack((torch.LongTensor(range(self.user_num)),torch.LongTensor(range(self.user_num)))).to(self.dev)
        all_edge = torch.cat([all_edge,I_edge],dim=1)
        all_val = torch.cat([all_val,torch.ones(self.user_num, device=self.dev)],dim=0)
        #构造新的稀疏邻接矩阵
        G = torch.sparse.FloatTensor(all_edge, all_val, adj.shape)
        G = self.norm_sparse_adj(G)         #标准化
        loss_ep = self.bce_criterion(G.to_dense(), norm_daj.to_dense())
        return G, loss_ep

    def graph_convolution(self, adj, X):
        item_embedding_final = [X]
        layer = X
        if self.Cgnn == 'sgc':
            for i in range(self.num_layer):
                layer = torch.matmul(adj, layer)
                item_embedding_final += [layer]
        else:
            layer = self.con_gnn(layer, adj)
            item_embedding_final += [layer]
        if self.con_layer_agg == 'sum':
            item_embedding_final = torch.sum(torch.stack(item_embedding_final,dim=-1),dim=-1)
        elif self.con_layer_agg == 'avg':
            item_embedding_final = torch.mean(torch.stack(item_embedding_final,dim=-1),dim=-1)
        elif self.con_layer_agg == 'concat':
            a = torch.stack(item_embedding_final,dim=-1)
            item_embedding_final = torch.cat(item_embedding_final,dim=-1)
            item_embedding_final = self.con_layer_w(item_embedding_final)
        else:
            item_embedding_final = item_embedding_final[-1] + item_embedding_final[0]
        return item_embedding_final

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)                            #纵向拼接i，j（512，64）
        if sim == 'cos':
            sim = torch.nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp                           #点乘计算相似度
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)       #（512，1）
        mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.aug_nce_fct(logits, labels)
        return loss

    def user_driven(self, user_ids, train):
        if train:
            if self.user_aug:
                adj = self.SparseDropout(self.user_adj_matrix)
                self.uvd_p, loss_ep = self.graph_augmentation(adj, self.user_emb.weight)    #user graph aug
            else:
                self.uvd_p = self.norm_sparse_adj(self.user_adj_matrix)
                loss_ep = 0
        else:
            loss_ep = 0
        uvd = self.graph_convolution(self.uvd_p, self.user_emb.weight)
        pos_u_embs = uvd[torch.LongTensor(user_ids).to(self.dev)]
        return pos_u_embs, loss_ep



    def session_encoder(self, sess):
        seq_emb = self.item_emb(torch.LongTensor(sess).to(self.dev))
        mask = (sess == 0)
        pref_item = self.encoder(seq_emb, mask)[:, -1, :]               
        return pref_item

    
    def forward(self, user_ids, log_seqs, sem_seq, pos_seqs): # for training
        pos_u_embs, loss_ep = self.user_driven(user_ids, train=True)               #user graph enhance
        ##session encoder
        pref_item = self.session_encoder(log_seqs)      
        #info_nce
        if self.user_cl:
            sem_log_feats = self.session_encoder(sem_seq)                               
            nce_loss = self.info_nce(pref_item, sem_log_feats, temp=1, batch_size=sem_log_feats.shape[0])       #NCEloss
            loss_ep += nce_loss * self.nu
        pos_feat = (1 - self.sigma) * pref_item + (self.sigma * pos_u_embs)       #user embedding + session embedding

        logits = torch.matmul(pos_feat, self.item_emb.weight.transpose(0, 1))
        loss = self.loss_fct(logits, torch.LongTensor(pos_seqs).to(self.dev))
        loss += loss_ep * self.xi
        return loss

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        pos_u_embs, _ = self.user_driven(user_ids, train=False)
        pos_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        pref_item = self.session_encoder(log_seqs)  # user_ids hasn't been used yet
        pos_feat = (1 - self.sigma) * pref_item + (self.sigma * pos_u_embs)
        logits = pos_embs.matmul(pos_feat.unsqueeze(-1)).squeeze(-1)
        return logits # preds # (U, I)
