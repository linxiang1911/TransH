file1="train.tsv"
file2="relation2id.txt"
file3="entity2id.txt"
num=1
relations_num=1
entities={}
relations={}
with open(file1, 'r') as f1:
        lines1 = f1.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 3:
                continue
            if line[0] not in entities:
                entities[line[0]]=str(num)
                num=num+1
            if line[1] not in relations:
                relations[line[1]]=str(relations_num)
                relations_num=relations_num+1
            if line[2] not in entities:
                entities[line[2]]=str(num)
                num=num+1
# for key in entities.keys():
#      print(key+" "+entities[key])
with open(file2, "w") as f_e:
        for e in relations.keys():
            f_e.write(e + "\t")
            f_e.write(relations[e])
            f_e.write("\n")
with open(file3, "w") as f_e:
        for e in entities.keys():
            f_e.write(e + "\t")
            f_e.write(entities[e])
            f_e.write("\n")

             