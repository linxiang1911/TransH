import json
import operator # operator模块输出一系列对应Python内部操作符的函数
import time
from tqdm import tqdm 
import numpy as np
import codecs
import datetime
import math

entity2id = {}
relation2id = {}
id2entity = {}
id2relation = {}
def test_data_loader(entity_embedding_file, norm_relation_embedding_file, hyper_relation_embedding_file,):
    print("load data...")
    file1 = entity_embedding_file
    file2 = norm_relation_embedding_file
    file3 = hyper_relation_embedding_file

    entity_dic = {}
    norm_relation = {}
    hyper_relation = {}

    with codecs.open(file1, 'r') as f1, codecs.open(file2, 'r') as f2, codecs.open(file3, 'r') as f3:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity_dic[line[0]] = json.loads(line[1])

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            norm_relation[line[0]] = json.loads(line[1])

        for line in lines3:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            hyper_relation[line[0]] = json.loads(line[1])

    print("Complete load. entity : %d , relation : %d " % (
        len(entity_dic), len(norm_relation)))
    
    return entity_dic, norm_relation, hyper_relation

class testTransH:
    def __init__(self, entities_dict, norm_relation, hyper_relation, relation,tail):
        self.entities = entities_dict
        self.norm_relation = norm_relation
        self.hyper_relation = hyper_relation
        self.mean_rank = 0
        self.hit_10 = 0
        self.hit_20 = 0
        self.relation=relation
        self.tail=tail
        self.t=10

    def test_run(self,recordname):
        rank_head_dict = {}
        tail = entity2id[self.tail]
        relation= relation2id[self.relation]
        for entity in self.entities.keys():
            head_triple = [entity, relation, tail]
            head_embedding = self.entities[head_triple[0]]
            tail_embedding = self.entities[head_triple[2]]
            norm_relation = self.norm_relation[head_triple[1]]
            hyper_relation = self.hyper_relation[head_triple[1]]
            distance = self.distance(head_embedding, norm_relation,hyper_relation, tail_embedding)
            rank_head_dict[tuple(head_triple)] = distance
        rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
        for i in range(10):
            print(id2entity[rank_head_sorted[i][0][0]])
            score=rank_head_sorted[i][1]
            p=math.exp(0-self.t*score)
            print(round(p, 3))

        return self.hit_10,self.hit_20, self.mean_rank


    def distance(self, h, r_norm, r_hyper, t):
        head = np.array(h)
        norm = np.array(r_norm)
        hyper = np.array(r_hyper)
        tail = np.array(t)
        h_hyper = head - np.dot(norm, head) * norm
        t_hyper = tail - np.dot(norm, tail) * norm
        d = h_hyper + hyper - t_hyper
        return np.sum(np.square(d))



if __name__ == "__main__":

    with codecs.open('semicon/entity2id.txt', 'r') as f1:
        content = f1.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            entity = triple[0]
            id = triple[1]
            entity2id[entity] = id
    with codecs.open('semicon/relation2id.txt', 'r') as f2:
        content = f2.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 2:
                continue
            relation = triple[0]
            id = triple[1]
            relation2id[relation] = id

    # 反转entity2id，得到id2entity
    id2entity = {v:k for k, v in entity2id.items()}

    # 反转relation2id，得到id2relation
    id2relation = {v:k for k, v in relation2id.items()}
    entity, norm_relation, hyper_relation= test_data_loader("weights_and_files/entity_dim50_nbatchs400_epoch40_loss1.027135",
                                                               "weights_and_files/rel_norm_dim50_nbatchs400_epoch40_loss1.027135",
                                                               "weights_and_files/rel_hyper_dim50_nbatchs400_epoch40_loss1.027135")

    #recordname = "dim: 200, nbatchs: 100, batchsize: 836, epoch: 500"
    recordname = "dim: 50, nbatchs: 400, batchsize: 836, epoch: 30"
    # test_triple: [id,id,id]  entity: { id: emb }
    # entity2id : { <id>: id } relation2id : { <id>: id }
    #have	toolset_indicator_spc:423
    relation = "have"
    tail = "toolset_indicator_spc:423"
    test = testTransH(entity, norm_relation, hyper_relation, relation,tail)
    test.test_run(recordname)
    # print("raw entity hits@10: ", hit10)
    # print("raw entity hits@20: ", hit20)
    # print("raw entity meanrank: ",mean_rank)