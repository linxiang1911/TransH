import bentoml
import json
import operator # operator模块输出一系列对应Python内部操作符的函数
import time
from tqdm import tqdm 
import numpy as np
import codecs
import datetime
import math
from bentoml.io import NumpyNdarray
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

class MyModel(bentoml.BentoService):
    def __init__(self):
        self.entities,self.norm_relation,self.hyper_relation  = test_data_loader("weights_and_files/entity_dim50_nbatchs400_epoch40_loss1.027135",
                                                               "weights_and_files/rel_norm_dim50_nbatchs400_epoch40_loss1.027135",
                                                               "weights_and_files/rel_hyper_dim50_nbatchs400_epoch40_loss1.027135")
        self.t=10

    @bentoml.api(input_type=bentoml.types.NumpyNdarrayInput(), output_type=bentoml.types.NumpyNdarrayOutput())
    def predict(input_series: np.ndarray)-> np.ndarray:
        print(input_series)
        result = np.random.rand(2, 3)
        return result


if __name__ == '__main__':
    # 创建服务类实例
    my_service = MyModel()

    # 将服务实例打包为BentoML bundle
    saved_path = my_service.save()

    # 将BentoML bundle加载到内存中
    loaded_service = bentoml.load(saved_path)

    # 使用已加载的服务启动REST API服务器
    api_server = loaded_service.get_service_api_server(port=5002)
    api_server.start()