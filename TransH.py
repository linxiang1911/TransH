#from py2neo import Graph,Node,Relationship, NodeMatcher 
from tqdm import tqdm 
import torch 
import torch.optim as optim
import torch.nn.functional as F
import random
import copy
import time
import codecs
import numpy as np
from tqdm import tqdm 
import json
# from sklearn.model_selection import train_test_split

entity2id = {}
relation2id = {}
loss_ls = []


def data_loader(file):
    file1 = file + "train.tsv"
    file2 = file + "entity2id.txt"
    file3 = file + "relation2id.txt"

    with open(file2, 'r') as f1, open(file3, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity2id[line[0]] = line[1]

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = line[1]

    entity_set = set()
    relation_set = set()
    triple_list = []

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[2]]
            r_ = relation2id[triple[1]]

            triple_list.append([h_, r_, t_])

            entity_set.add(h_)
            entity_set.add(t_)

            relation_set.add(r_)

    return entity_set, relation_set, triple_list

      

class TransH():
    def __init__(self, entity_set, relation_set, triple_list, embedding_dim = 50, lr = 0.01, margin = 1.0, C = 1.0):
        self.entities = entity_set#{id:emb}
        self.relations = relation_set#{id:emb}
        self.triples = triple_list#[id,id,id]
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.loss = 0.0
        self.norm_relations = {}
        self.hyper_relations = {}#{id:emb}
        self.C = C
    def data_initialise(self, continue_trian = 0, entity_name = "None", rel_hyper_name = "None", rel_norm_name = "None"):
        if continue_trian == 0:
            entityVectorList = {}
            relationNormVectorList = {}
            relationHyperVectorList = {}

            for entity in self.entities:
                entity_vector = torch.Tensor(self.dimension).uniform_(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension))
                entityVectorList[entity] = entity_vector.requires_grad_(True)

            for relation in self.relations:
                relation_norm_vector = torch.Tensor(self.dimension).uniform_(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension))
                relation_hyper_vector = torch.Tensor(self.dimension).uniform_(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension))

                relationNormVectorList[relation] = relation_norm_vector.requires_grad_(True)
                relationHyperVectorList[relation] = relation_hyper_vector.requires_grad_(True)

            self.entities = entityVectorList#{id:emb}
            self.norm_relations = relationNormVectorList
            self.hyper_relations = relationHyperVectorList
        else:
            entity_dic = {}
            rel_hyper_dic = {}
            rel_norm_dic = {}
            with codecs.open(entity_name, 'r') as f1, codecs.open(rel_hyper_name, 'r') as f2, codecs.open(rel_norm_name, 'r') as f3:
                content1 = f1.readlines()
                content2 = f2.readlines()
                content3 = f3.readlines()
                for line in content1:
                    line = line.strip().split("\t")
                    if len(line) != 2:
                        continue
                    entity_dic[int(line[0])] = torch.Tensor(json.loads(line[1])).requires_grad_(True)
                for line in content2:
                    line = line.strip().split("\t")
                    if len(line) != 2:
                        continue
                    rel_hyper_dic[int(line[0])] = torch.Tensor(json.loads(line[1])).requires_grad_(True)
                for line in content3:
                    line = line.strip().split("\t")
                    if len(line) != 2:
                        continue
                    rel_norm_dic[int(line[0])] = torch.Tensor(json.loads(line[1])).requires_grad_(True)
            self.entities = entity_dic
            self.norm_relations = rel_norm_dic
            self.hyper_relations = rel_hyper_dic
                    
            

    def training_run(self, epochs=100, times = 2 ,nbatches = 50):
        batch_size = int(len(self.triples) / nbatches)
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0.0

            for batch in tqdm(range(nbatches),ncols=80):
                batch_samples = random.sample(self.triples, batch_size)#[id,id,id]
                Tbatch = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    seed = random.random()
                    if seed < 0.5:
                        # 更改头节点
                        corrupted_sample[0] = random.sample(self.entities.keys(), 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities.keys(), 1)[0]
                    else:
                        # 更改尾节点
                        corrupted_sample[2] = random.sample(self.entities.keys(), 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entities.keys(), 1)[0]

                    if (sample, corrupted_sample) not in Tbatch:
                        Tbatch.append((sample, corrupted_sample))

                self.update_triple_embedding(Tbatch)
            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            
            print("running loss: ", self.loss/len(self.triples))
            loss_ls.append(self.loss)

        with codecs.open("weights_and_files/entity_dim" + str(self.dimension) + "_nbatchs" + str(nbatches) + "_epoch" + str(epochs*times) + "_loss"+ str(self.loss/len(self.triples))[0:8], "w") as f1:
            print("写f1")
            for e in tqdm(self.entities,ncols=80):
                f1.write(str(e) + "\t")
                f1.write(str(list(self.entities[e].detach().numpy())))
                f1.write("\n")

        with codecs.open("weights_and_files/rel_norm_dim" + str(self.dimension) + "_nbatchs" + str(nbatches) + "_epoch" + str(epochs*times) + "_loss"+ str(self.loss/len(self.triples))[0:8], "w") as f2:
            print("写f2")
            for r in tqdm(self.norm_relations,ncols=80):
                f2.write(str(r) + "\t")
                f2.write(str(list(self.norm_relations[r].detach().numpy())))
                f2.write("\n")

        with codecs.open("weights_and_files/rel_hyper_dim" + str(self.dimension) + "_nbatchs" + str(nbatches) + "_epoch" + str(epochs*times) + "_loss"+ str(self.loss/len(self.triples))[0:8], "w") as f3:
            print("写f3")
            for r in tqdm(self.hyper_relations,ncols=80):
                f3.write(str(r) + "\t")
                f3.write(str(list(self.hyper_relations[r].detach().numpy())))
                f3.write("\n")
        with codecs.open("loss", "w") as f3:
            f3.write(str(loss_ls))

    def norm_l2(self, h, r_norm, r_hyper, t):
        return torch.norm(h - r_norm.dot(h)*r_norm + r_hyper -(t - r_norm.dot(t)*r_norm))

    # loss = F.relu(self.margin + correct_distance - corrupted_distance) + self.C * scale
    # 模长约束
    def scale_entity(self, vector):
        return torch.relu(torch.sum(vector**2) - 1)

    def update_triple_embedding(self, Tbatch):
        for correct_sample, corrupted_sample in Tbatch:
            correct_head = self.entities[correct_sample[0]]#entities:{id:emb}
            correct_tail  = self.entities[correct_sample[2]]
            relation_norm = self.norm_relations[correct_sample[1]]
            relation_hyper = self.hyper_relations[correct_sample[1]]

            corrupted_head = self.entities[corrupted_sample[0]]
            corrupted_tail = self.entities[corrupted_sample[2]]

            opt1 = optim.SGD([correct_head], lr=self.learning_rate)
            opt2 = optim.SGD([correct_tail], lr=self.learning_rate)
            opt3 = optim.SGD([relation_norm], lr=self.learning_rate)
            opt4 = optim.SGD([relation_hyper], lr=self.learning_rate)

            if correct_sample[0] == corrupted_sample[0]:
                opt5 = optim.SGD([corrupted_tail], lr=self.learning_rate)
                correct_distance = self.norm_l2(correct_head, relation_norm, relation_hyper, correct_tail)
                corrupted_distance = self.norm_l2(correct_head, relation_norm, relation_hyper, corrupted_tail)
                scale = self.scale_entity(correct_head) + self.scale_entity(correct_tail) + self.scale_entity(corrupted_tail)

            else:
                opt5 = optim.SGD([corrupted_head], lr=self.learning_rate)
                correct_distance = self.norm_l2(correct_head, relation_norm, relation_hyper, correct_tail)
                corrupted_distance = self.norm_l2(corrupted_head, relation_norm, relation_hyper, correct_tail)
                scale = self.scale_entity(correct_head) + self.scale_entity(correct_tail) + self.scale_entity(
                    corrupted_head)

            opt1.zero_grad()
            opt2.zero_grad()
            opt3.zero_grad()
            opt4.zero_grad()
            opt5.zero_grad()

            loss = F.relu(self.margin + correct_distance - corrupted_distance)
            loss.backward()
            self.loss += loss.item()
            opt1.step()
            opt2.step()
            opt3.step()
            opt4.step()
            opt5.step()


            self.entities[correct_sample[0]] = correct_head #
            self.entities[correct_sample[2]] = correct_tail
            if correct_sample[0] == corrupted_sample[0]:
                self.entities[corrupted_sample[2]] = corrupted_tail
            elif correct_sample[2] == corrupted_sample[2]:
                self.entities[corrupted_sample[0]] = corrupted_head
            self.norm_relations[correct_sample[1]] = relation_norm
            self.hyper_relations[correct_sample[1]] = relation_hyper


if __name__ == '__main__':
    file1 = "./openbiolink/"
    entity_set, relation_set, triple_list = data_loader(file1)
    transH = TransH(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=1.0)
    #continue_train表示是否从之前的训练模型继续训练，后面是文件名称
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))
    # transH.data_initialise(continue_trian = 1, 
    #                         entity_name = "weights_and_files/entity_dim200_nbatchs20_epoch400_loss0.003410", 
    #                         rel_hyper_name = "weights_and_files/rel_hyper_dim200_nbatchs20_epoch400_loss0.003410", 
    #                         rel_norm_name = "weights_and_files/rel_norm_dim200_nbatchs20_epoch400_loss0.003410")
    transH.data_initialise()
    transH.training_run(epochs=200,times=1,nbatches=400)#times表示第几次训练，跟最后生成的参数文件名有关
    
    

    # 下面这部分代码是生成relation2id和entity2id的，如果neo4j重新导入过，这个要执行一边
    # with codecs.open("relation2id", "w") as f2:
    #     print("写relation2id")
    #     for item  in tqdm(relation2id.items(),ncols=80):
    #         f2.write(str(item[0]) + "\t" + str(item[1]))
    #         f2.write("\n")
    # with codecs.open("entity2id", "w") as f2:
    #     print("写entity2id")
    #     for item  in tqdm(entity2id.items(),ncols=80):
    #         f2.write(str(item[0]) + "\t" + str(item[1]))
    #         f2.write("\n")
