with open('test.txt', 'r') as file1:
    data1_content = file1.read()

with open('test_data.txt', 'r') as file2:
    data2_content = file2.read()

# 去除第一个文件和第二个文件内容的多余换行符
data1_content = data1_content.rstrip('\n')
data2_content = data2_content.lstrip('\n')

# 将第一个文件和第二个文件内容合并，并在它们之间插入一个换行符
combined_content = data1_content + '\n' + data2_content

# 将合并后的内容写回到第一个文件
with open('test.txt', 'w') as file1:
    file1.write(combined_content)