import networkx as nx

# 创建一个空的有向图
G = nx.DiGraph()

# 从文件中读取数据并添加边
with open("testOLD.txt", "r") as file:
    for line in file:
        # 分割行以获取节点信息
        nodes = line.strip().split()

        # 第一个节点是起始节点，后面的节点都是与之交互的节点
        source_node = nodes[0]
        target_nodes = nodes[1:]

        # 添加边
        for target_node in target_nodes:
            G.add_edge(source_node, target_node)

# 获取节点数量
node_count = len(G.nodes())

# 打印节点数量
print("节点数量：", node_count)
