#Merges labels into only 1 dict based on obj name

import json
Indexing = None
with open("Indexing.json", "r") as f:
    Indexing = json.load(f)

with open("shapes_test.json", "r") as f:
    shapes_test = json.load(f)

with open("shapes_train.json", "r") as f:
    shapes_train = json.load(f)

with open("cultures_train.json", "r") as f:
    cultures_train = json.load(f)

with open("cultures_test.json", "r") as f:
    cultures_test = json.load(f)

#Dict with index key: obj_name, value: label1+label2
Obj_Index = {}

for i in Indexing:
    label1 = ""
    label2 = ""
    if Indexing[i][0] != None:
        label1 = shapes_train[Indexing[i][0]]
    if Indexing[i][1] != None:
        label1 = shapes_test[Indexing[i][1]]
    if Indexing[i][2] != None:
        label2 = cultures_train[Indexing[i][2]]
    if Indexing[i][3] != None:
        label2 = cultures_test[Indexing[i][3]]
    Obj_Index[i] = label1 + label2

with open("obj_indexer.json", "w") as f:
    json.dump(Obj_Index, f)
