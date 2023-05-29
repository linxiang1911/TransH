import random

def process_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        # 去掉行末换行符
        line = line.strip()
        # 将三元组用\t分隔开，并在最后加入一个\t
        elems = line.split('\t')
        new_line = '\t'.join(elems) + '\t'
        # 生成一个0.5到1之间的小数，并保留一位小数
        random_float = round(random.uniform(0.5, 1), 1)
        # 将生成的小数追加到新行的结尾处
        new_line += str(random_float) + '\n'
        new_lines.append(new_line)
    # 将处理后的行写回文件
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

# 处理train.txt和valid.txt两个文件
process_file('./semicon/train.txt')