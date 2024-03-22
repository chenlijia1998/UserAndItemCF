from sklearn.metrics import normalized_mutual_info_score
import networkx as nx
import community
# 读取数据集文件
dataset_file = "111.txt"
dataset = []
print(dataset_file)
with open(dataset_file, 'r') as file:
    for count,line in enumerate(file):
        if count >= 2:
            break
        user, item = line.strip().split(' ')
        dataset.append([int(user), int(item)])

# 统计物品之间的共现次数
item_cooccurrence = {}
for interaction in dataset:
    user, item = interaction
    if user not in item_cooccurrence:
        item_cooccurrence[user] = set()
    item_cooccurrence[user].add(item)

# 创建加权网络图
G = nx.Graph()
for user_items in item_cooccurrence.values():
    items = list(user_items)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            item_pair = (items[i], items[j])
            if item_pair in G.edges:
                G.edges[item_pair]['weight'] += 1
            else:
                G.add_edge(item_pair[0], item_pair[1], weight=1)
# 保存图
filename = "network_data999.txt"
nx.write_weighted_edgelist(G, filename)
# 社区检测
partition = community.best_partition(G)
# 构建节点和社区ID的映射关系
community_mapping = {}
for node, community_id in partition.items():
    community_mapping[node] = community_id
# 输出每个节点及其所属的社区
for node, community_id in community_mapping.items():
    print(f"Node {node}: Community {community_id}")
# 保存社区
with open('community.txt', 'w') as file:
    for node, community_id in community_mapping.items():
        line = f"{node}\t{community_id}\n"
        file.write(line)


