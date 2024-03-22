import random
# 打开文件并读取数据
with open('community.txt', 'r') as file:
    lines = file.readlines()

# 初始化最大值为负无穷
max_value = float('-inf')

# 遍历数据集，找到第一列的最大值
for line in lines:
    columns = line.strip().split()
    if len(columns) >= 2:
        first_column = int(columns[0])
        if first_column > max_value:
            max_value = first_column

# 将第二列的每个数字加上最大值
result = []
for line in lines:
    columns = line.strip().split()
    if len(columns) >= 2:
        first_column = int(columns[0])
        second_column = int(columns[1])
        new_second_column = second_column + max_value
        result.append(f'{first_column} {new_second_column}\n')

# 将结果写回文件
with open('output.txt', 'w') as output_file:
    output_file.writelines(result)
# 打开处理后的文件
with open('output.txt', 'r') as file:
    lines = file.readlines()

# 创建一个空字典来跟踪节点之间的交互关系
interactions = {}

# 遍历处理后的数据
for line in lines:
    columns = line.strip().split()
    if len(columns) >= 2:
        first_node = int(columns[0])
        second_node = int(columns[1])

        # 更新字典，将第二列节点作为键，将与其交互的第一列节点添加到值列表中
        if second_node not in interactions:
            interactions[second_node] = []
        interactions[second_node].append(first_node)

# 构建新的数据结构，将交互过的节点放在同一行
result_data = []
for second_node, first_nodes in interactions.items():
    line = f"{second_node} {' '.join(map(str, first_nodes))}\n"
    result_data.append(line)

# 将结果写回文件
with open('interactions.txt', 'w') as output_file:
    output_file.writelines(result_data)

# 打开包含交互数据的文件
with open('interactions.txt', 'r') as file:
    lines = file.readlines()

# 创建字典来保存每个第一列节点的交互节点
node_interactions = {}

# 遍历交互数据，将每个第一列节点的交互节点添加到字典中
for line in lines:
    columns = line.strip().split()
    if len(columns) >= 2:
        second_node = int(columns[0])
        first_nodes = list(map(int, columns[1:]))
        node_interactions[second_node] = first_nodes

# 创建用于划分数据的字典
train_data = {}
test_data = {}

# 设置划分比例
split_ratio = 0.8  # 80%用于训练，20%用于测试

# 遍历每个第一列节点的交互节点
for second_node, first_nodes in node_interactions.items():
    # 随机划分第二列节点
    random.shuffle(first_nodes)
    split_index = int(len(first_nodes) * split_ratio)

    # 将数据划分为训练和测试集
    train_data[second_node] = first_nodes[:split_index]
    test_data[second_node] = first_nodes[split_index:]

# 将训练数据保存到文件
with open('train_data.txt', 'w') as train_file:
    for second_node, first_nodes in train_data.items():
        line = f"{second_node} {' '.join(map(str, first_nodes))}\n"
        train_file.write(line)

# 将测试数据保存到文件
with open('test_data.txt', 'w') as test_file:
    for second_node, first_nodes in test_data.items():
        line = f"{second_node} {' '.join(map(str, first_nodes))}\n"
        test_file.write(line)