import networkx as nx
import community
from sklearn.metrics import normalized_mutual_info_score
dataset_file = r"user_ratings.txt"
item_cooccurrence = {}

with open(dataset_file, 'r') as file:
    for count,line in enumerate(file):
        if count >= 900000:
            break
        data = line.strip().split(',')
        user = data[0]
        items = data[1:]

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item_pair = (items[i], items[j])
                if item_pair in item_cooccurrence:
                    item_cooccurrence[item_pair] += 1
                else:
                    item_cooccurrence[item_pair] = 1

G = nx.Graph()
for item_pair, weight in item_cooccurrence.items():
    item1, item2 = item_pair
    G.add_edge(item1, item2, weight=weight)

filename = "M_network_data.txt"
nx.write_weighted_edgelist(G, filename)
print("end")
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
with open('M_community.txt', 'w') as file:
    for node, community_id in community_mapping.items():
        line = f"{node}\t{community_id}\n"
        file.write(line)

# 打印模块度指标
modularity = community.modularity(partition, G)
print(f"Modularity: {modularity}")
# 计算每个社区的密度
for community_id in set(partition.values()):
    nodes_in_community = [node for node, cid in partition.items() if cid == community_id]
    subgraph = G.subgraph(nodes_in_community)
    num_edges = subgraph.number_of_edges()
    num_possible_edges = len(nodes_in_community) * (len(nodes_in_community) - 1) / 2  # 完全图情况下的边数
    density = num_edges / num_possible_edges
    print(f"Community {community_id} density: {density}")