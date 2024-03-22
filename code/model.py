"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
import math
from dataloader import BasicDataset
import dataloader
from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle

# 读取文件
with open('community.txt', 'r') as file:
    lines = file.readlines()

# 创建一个空字典，用于存储社区集合
community_dict = {}

# 遍历文件的每一行
for line in lines:
    # 按逗号分隔每一行，获取节点和社区信息
    node, community = line.strip().split()
    # 将社区信息转换为整数类型
    community = int(community)
    # 检查社区是否已经在字典中
    if community in community_dict:
        # 如果在，将节点添加到对应社区的集合中
        community_dict[community].append(int(node))
    else:
        # 如果不在，创建一个新的社区集合，并将节点添加进去
        community_dict[community] = [int(node)]

# 重新排列字典按照社区顺序
community_dict = dict(sorted(community_dict.items()))
# 假设你已经有了community_dict字典，其中键为社区编号，值为节点列表
# 创建一个张量矩阵，行数为社区的个数，列数为节点的个数
num_communities = len(community_dict)
num_nodes = max(max(nodes) for nodes in community_dict.values()) + 1
tensor_comunity_matrix = torch.zeros(num_communities, num_nodes)

# 根据节点属于的社区，在对应的位置上将元素设置为1
for community_idx, nodes in community_dict.items():
    for node in nodes:
        tensor_comunity_matrix[community_idx, node] = 1



class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

    
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["hidden_size"] % config[
            "num_of_attention_heads"] == 0, "The hidden size is not a multiple of the number of attention heads"

        self.num_attention_heads = config['num_of_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / config['num_of_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
                                                                         -2))  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs,
                                     value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (
        self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)

        return attention_probs
    
class SelfAttentionLayer(nn.Module):
    
    def __init__(self, dim_input, dim_q, dim_v):
        '''
        参数说明：
        dim_input: 输入数据x中每一个样本的向量维度
        dim_q:     Q矩阵的列向维度, 在运算时dim_q要和dim_k保持一致;
                   因为需要进行: K^T*Q运算, 结果为：[dim_input, dim_input]方阵
        dim_v:     V矩阵的列项维度,维度数决定了输出数据attention的列向维度
        '''
        super(SelfAttentionLayer, self).__init__()

        # dim_k = dim_q
        self.dim_input = dim_input
        self.dim_q = dim_q
        self.dim_k = dim_q
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(self.dim_input, self.dim_q, bias=False)
        self.linear_k = nn.Linear(self.dim_input, self.dim_k, bias=False)
        self.linear_v = nn.Linear(self.dim_input, self.dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(self.dim_k)

    def forward(self, x):
        batch, n, dim_q = x.shape

        q = self.linear_q(x)  # Q: batch_size * seq_len * dim_k
        k = self.linear_k(x)  # K: batch_size * seq_len * dim_k
        v = self.linear_v(x)  # V: batch_size * seq_len * dim_v
#         print(f'x.shape:{x.shape} \n  Q.shape:{q.shape} \n  K.shape: {k.shape} \n  V.shape:{v.shape}')
        # K^T*Q
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        # 归一化获得attention的相关系数：A
        dist = torch.softmax(dist, dim=-1)
#         print(dist)
#         print('attention matrix: ', dist.shape)
        # socre与v相乘，获得最终的输出
        att = torch.bmm(dist, v)
        # print('attention output: ', att.shape)
        return dist
    
    
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        self.user_similarity = None  # 保存社区用户相似度的列表
        # 创建SelfAttentionLayer实例
#         self.self_attention = SelfAttentionLayer(dim_input, dim_q, dim_v)  
        self.multi_self_attention = BertSelfAttention(config)
        
    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.weight_att = nn.Parameter(torch.empty(1).uniform_(0, 0.001), requires_grad=True)
        self.weight_b = nn.Parameter(torch.empty(1).uniform_(0, 1), requires_grad=True)
#         self.weight_att = nn.Parameter(torch.randn(1), requires_grad=True) # 定义一个可学习的参数
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph
        # print(g_droped.shape)
        # print(all_emb.shape)
        all_emb = torch.sparse.mm(g_droped, all_emb)
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
#         v1 = rating * self.weight_b
#         w_att = self.weight_att  # 系数参数
#         community_attentions = self.compute_user_attention(users)
#         # 删除每个注意力张量中的第一个元素，自身的权重
#         modified_tensors = [tensor[:, 1:] for tensor in community_attentions]
#         # 使用torch.cat将列表中的张量合并为一个张量
#         merged_attentions_tensor = torch.cat(modified_tensors, dim=0)
#         aw = merged_attentions_tensor * w_att
#         exp_att_aw = torch.exp(aw)
#         result_matmul = torch.matmul(exp_att_aw.to(world.device), tensor_comunity_matrix.to(world.device))
#         result_mul_rating = torch.mul(result_matmul, rating)
#         v2 = result_mul_rating
#         rating_end=self.f(v1+(1-self.weight_b)*v2)
#         return rating_end
        return rating
        
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
    def compute_user_attention(self,user):
        all_users, all_items = self.computer()
#         print(all_items)
#         print("all_items-----")
        user_emb = all_users[user]
#         community_users = all_users[-7:]
#         print(community_users)
#         print("community_users------")
        result = all_items.to(world.device).unsqueeze(0).repeat(tensor_comunity_matrix.to(world.device).size(0), 1, 1) * tensor_comunity_matrix.to(world.device).unsqueeze(2).float()
        community_users = torch.mean(result, dim=1)
#         print("community_users1-----")
#         print(community_users)
#         print("community_users2-----")
        user_attention = []
#         print(user_attention)
#         print("user_attention")
        for user in user_emb:
            user = user.unsqueeze(0)  # 增加一个维度
            input_vector = torch.cat((user, community_users), dim=0)
#             print("input_vector.unsqueeze(0)---------------------------")
#             print(input_vector.unsqueeze(0))
#             print("input_vector.unsqueeze(0)-------------------------------")
            attention_weights = self.multi_self_attention(input_vector.unsqueeze(0))  # Update to use the output of self_attention
            attention_weights = attention_weights.mean(dim=1)
            user_attention.append(attention_weights[0, 0, :].unsqueeze(0)) # 只保存第一行
#             print(user_attention)
#             print("user_attention-------------------------")

        return user_attention
    
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        print('forward被调用---------')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
         # compute attention weights
        user_attention = self.compute_user_attention()
        
        return gamma
      
    def get_user_similarity_old(self, user_list1, user_list2):
        print(user_list1)
        print(user_list2)
        print("user_old---------")
        all_users, all_items = self.computer()
        user_emb1 = all_users[user_list1]
        user_emb2 = all_users[user_list2]
        user_similarity = []
        for u1 in user_emb1:
            similarity = F.cosine_similarity(u1.unsqueeze(0), user_emb2, dim=1)
            similarity = F.normalize(similarity, dim=0)
            user_similarity.append(similarity)
        return user_similarity
      
    def get_user_similarity(self, user):
        # print(user)
        if self.user_similarity is None:
            self.compute_user_similarity()  # 首次调用时计算用户相似度
        similarities = []
        for u in user:
            similarity = self.user_similarity[u]
            similarities.append(similarity)
        return similarities


    def compute_user_similarity(self):
        all_users, all_items = self.computer()
        user_emb = all_users  # 获取除了社区用户之外的所有用户的嵌入向量
        community_users = all_users[-7:]  # 获取末尾7个社区用户的嵌入向量
        self.user_similarity = []

        for user in user_emb:
            similarities = F.cosine_similarity(user.unsqueeze(0), community_users, dim=1)
            similarities = F.normalize(similarities, dim=0)
            self.user_similarity.append(similarities.tolist())
        # print(self.user_similarity)
        # print("user_similarity-------------")
              # 将相似度列表扁平化以便绘图
#         flat_similarities = [sim for sublist in self.user_similarity for sim in sublist]

#         # 绘制相似度的直方图
#         plt.hist(flat_similarities, bins=200, edgecolor='black')
#         plt.title('Distribution')
#         plt.xlabel('Sim_Value')
#         plt.ylabel('rate')

#         # 保存图像
#         plt.savefig('user_similarity.png')

#         plt.show()
#      #   保存相似度数据
#         with open('user_similarity.pkl', 'wb') as f:
#             pickle.dump(self.user_similarity, f)

    
