# with open("train.tsv", 'r') as f1:
#         lines1 = f1.readlines()
#         for line in lines1:
#             line = line.strip().split('\t')
#             print(line)
import torch
print(torch.__version__)
num=[1,2,3]
strnum=str(num)
newnum=eval(strnum)
print(type(newnum))