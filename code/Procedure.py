'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import math
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import random
from torch.nn.utils.rnn import pad_sequence

# 读取文件，假设文件名为 'your_file.csv'，以逗号分隔
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
# 打印社区集合
# for community, nodes in community_dict.items():
#     print(f"Community {community}: {nodes}")
# print(community_dict)
# 假设你已经有了community_dict字典，其中键为社区编号，值为节点列表
# 创建一个张量矩阵，行数为社区的个数，列数为节点的个数
num_communities = len(community_dict)
num_nodes = max(max(nodes) for nodes in community_dict.values()) + 1
tensor_comunity_matrix = torch.zeros(num_communities, num_nodes)

# 根据节点属于的社区，在对应的位置上将元素设置为1
for community_idx, nodes in community_dict.items():
    for node in nodes:
        tensor_comunity_matrix[community_idx, node] = 1
tensor_comunity_matrix=tensor_comunity_matrix.to(world.device)

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, diversity = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))

    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def process_ratings_vectorized(rating_tensor, community_dict):
    num_users, num_items = rating_tensor.shape
    num_communities = len(community_dict)

    # 初始化一个新的张量，每一行表示一个用户对每个社区的评分
    processed_ratings = torch.zeros((num_users, num_communities, num_items), dtype=torch.float)

    # 获取用户评分，并将其转换为 Float 类型
    user_ratings_float = rating_tensor.float()

    # 创建一个索引张量，用于将评分放到对应的社区位置
    index_tensor = torch.zeros((num_users, num_communities, num_items), dtype=torch.long)

    for community_idx, community_items in community_dict.items():
        index_tensor[:, community_idx, community_items] = 1
    # 使用广播操作将评分放到对应的社区位置
    processed_ratings = user_ratings_float.unsqueeze(1).to(world.device) * index_tensor.float().to(world.device)
    return processed_ratings

def topk2D(
    input: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    max_k = k.max().int()
    fake_indexes = torch.topk(input, max_k, dim=1).indices

    T = torch.arange(max_k).expand_as(fake_indexes).to(world.device)
    T = torch.remainder(T, k.unsqueeze(1)).to(world.device)

    indexes = torch.gather(fake_indexes, 1, T).to(world.device)

    return indexes
    
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    max_K=max_K - 2

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'ils': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        communities_users = users[-7:]
        print(communities_users)
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list_1 = []
        rating_list = []
        groundTrue_list = []
        auc_record = []
        ils_list = []  # 用于存储每个用户的ILS
        ratings = []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)            
            ratings_end =[] #新增方法结果
            communitys_attentions = Recmodel.compute_user_attention(batch_users_gpu)

            with open('00306community_attention.txt','a') as file3:
                print(communitys_attentions,file=file3)
            
            # print(minicommunity_rate)
            selected_items = []  # 用于存储当前用户选择的项目
            commnities_result = []  # 用于存储结果，从selected_items规则选择的
            communities_end =[]  #最后结果
            CRatings=[] #按照社区划分过的评分列表
            
            #把一个用户对物品的评分按照物品的社区划分
            split_ratings=process_ratings_vectorized(rating,community_dict)
            #降维堆叠
            c_split_ratings=split_ratings.view(-1,split_ratings.size(-1)).to(world.device)
            # 删除每个注意力张量中的第一个元素，自身的权重
            modified_tensors = [tensor[:, 1:] for tensor in communitys_attentions]
            # 使用torch.cat将列表中的张量合并为一个张量
            merged_attentions_tensor = torch.cat(modified_tensors, dim=0)
            # 将原始张量展平为一行
            flattened_attentions_tensor = merged_attentions_tensor.view(1, -1)
            # 将张量中的每个元素乘以20并向上取整
            k_tensor = torch.ceil(flattened_attentions_tensor * 8).to(torch.int)
            # 使用torch.squeeze将张量降维
            att_k_tensor = torch.squeeze(k_tensor).to(world.device)
            rat=topk2D(c_split_ratings,att_k_tensor)
            c_rat_list=rat.tolist()
            # 将每七行的子列表合并为一个新的列表
            ratings_end = [sum(c_rat_list[i:i + 7], []) for i in range(0, len(c_rat_list), 7)]
            
            for sublist in ratings_end:
                one_community_end = random.sample(sublist, 2)
                communities_end.append(one_community_end)
            communities_end = torch.tensor(communities_end)
            communities_end = communities_end.int()

            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, max_K)
            rating = rating.cpu().numpy()
            aucs = [
                    utils.AUC(rating[i],
                              dataset,
                              test_data) for i, test_data in enumerate(groundTrue)
                ]
            auc_record.extend(aucs)
            del rating

            users_list.append(batch_users)
#             rating_list.append(rating_K.cpu())
            rating_list_1.append(rating_K.cpu())
            result = torch.cat((rating_list_1[-1], communities_end), dim=1)
            rating_list.append(result)
            groundTrue_list.append(groundTrue)
            
            for user_ratings in rating_list[-1]:
            # 将tensor转换为numpy数组
                recommended_item_indices = user_ratings.cpu().numpy()
#                 print(recommended_item_indices)
#                 print("recommended_item_indices---------")
                # 通过索引获取物品的嵌入向量
                recommended_items = []
                for index in recommended_item_indices:
#                     print(index)
#                     print("index")
                    item_embedding = Recmodel.embedding_item.weight[index].cpu().numpy() # 根据索引获取物品的嵌入向量
#                     print(item_embedding)
#                     print("item_embedding----")
                    recommended_items.append(item_embedding)

                # 计算前20个推荐物品的ILS
                ils = utils.ILS(recommended_items)

                ils_list.append(ils)

        assert total_batch == len(users_list)
        # 创建一个空列表来存储所有的recommended_item_indices
        all_recommended_item_indices = []
        for i, user_ratings in enumerate(rating_list):
            # 将每个张量转换为 NumPy 数组
            recommended_item_indices = user_ratings.cpu().numpy()
    
            # 将recommended_item_indices添加到列表中
            all_recommended_item_indices.append(recommended_item_indices)
            
            # 将所有的recommended_item_indices连接成一个数组
        all_recommended_item_indices = np.concatenate(all_recommended_item_indices)

        # 将数组保存到一个文本文件中
        np.savetxt("00306edges.txt", all_recommended_item_indices, delimiter=",", fmt='%d')
        print("---------")
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['ils'] = np.mean(ils_list)
        results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        with open('00306result.txt','a') as file1:
            print(results,file=file1)
        print(results)
        return results
