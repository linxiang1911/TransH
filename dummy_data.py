from ordered_set import OrderedSet
import pandas as pd
import os
disease2bin={}
binnum=1
food2etest={}
etestnum=1
medicine2sympAndmedi={}
df = pd.DataFrame(columns=['source', 'relation', 'target'])
all_df = pd.DataFrame(columns=['source', 'relation', 'target'])
medicine = OrderedSet()
for split in ['train', 'test', 'valid']:
    for line in open('./medical/{}.txt'.format(split)):
        line  =  line.strip().split('\t')
    # if len(list)!=3:
    #     print(line)
    #     print(list)
        #print(list)
        if line[0] not in disease2bin:
            disease2bin[line[0]]='bin:'+str(binnum)
            binnum=binnum+1
        if line[1]=='宜吃':
            if line[2] not in food2etest:
                food2etest[line[2]]='etest:'+str(etestnum)
                etestnum=etestnum+1
            etestItem=food2etest[line[2]]
            new_row = {'source': disease2bin[line[0]], 'relation': 'pass', 'target':etestItem}
            all_df = all_df.append(new_row, ignore_index=True)
        if line[1]=='忌吃':
            if line[2] not in food2etest:
                food2etest[line[2]]='etest:'+str(etestnum)
                etestnum=etestnum+1
            etestItem=food2etest[line[2]]
            new_row = {'source': disease2bin[line[0]], 'relation': 'fail', 'target':etestItem}
            all_df = all_df.append(new_row, ignore_index=True)
        if line[1]=='好评药品':
            new_row = {'source': line[0], 'relation': line[1], 'target':line[2]}
            medicine.add(line[2])
            df = df.append(new_row, ignore_index=True)

medicine_num=len(medicine)//2
medicine_list=list(medicine)   
for i in range(medicine_num):
   medicine2sympAndmedi[medicine_list[i]]='defect:'+str(i+1) 
for i in range(medicine_num,len(medicine)):
   medicine2sympAndmedi[medicine_list[i]]='toolset_indicator_spc:' +str(i-medicine_num+1)

for index, row in df.iterrows(): 
    source = row['source']
    relation = row['relation']
    target = row['target'] 
    target= medicine2sympAndmedi[target]
    if target.startswith('defect'):
        relation='need_check'
    if target.startswith('toolset_indicator_spc'):
        relation='have'
    new_row= {'source': disease2bin[source], 'relation': relation, 'target':target}   
    all_df = all_df.append(new_row, ignore_index=True)

print(all_df.head)
os.makedirs('./semicon', exist_ok=True)
all_df = all_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 划分训练集、验证集和测试集
n = len(all_df)
train_split = int(n * 0.9)

train_df = all_df[:train_split]
test_df = all_df[train_split:]

# train_split = int(n * 0.8)
# valid_split = int(n * 0.9)

# train_df = all_df[:train_split]
# valid_df = all_df[train_split:valid_split]
# test_df = all_df[valid_split:]

# 存储三个文件
train_df.to_csv('./semicon/train.txt', sep='\t', index=False, header = False)
test_df.to_csv('./semicon/test.txt', sep='\t', index=False, header = False)