# 读取原始TXT文件
file_path = 'user_ratings.txt'  # 请替换成你的文件路径
with open(file_path, 'r') as file:
    lines = file.readlines()
print("start")
# 在每一行的开头添加索引
lines_with_index = [f'{index} {line}' for index, line in enumerate(lines)]

# 将带有索引的行写回文件
with open(file_path, 'w') as file:
    file.writelines(lines_with_index)
print("end----")