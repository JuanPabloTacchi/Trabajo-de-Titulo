import os
from datasetCreationUtils import getObjName
#This script combines the indexes of the dataset (based on the .obj metadata) into a joint set for further identification


#all paths must be the original mesh folders
#add your own paths here
path_shape_train = ".../shape/train"
path_shape_test = ".../shape/test"
path_culture_train = ".../culture/train"
path_culture_test = ".../culture/test"

extensions_3d = ('.obj')

#Dict key (texture_name): [index_shape_train, index_shape_test, index_culture_train, index_culture_test]
Indexing  = {}

print("Starting")
print("Indexing shape train")
path = path_shape_train
for obj in os.listdir(path):
        if obj.endswith(extensions_3d):
            path_obj = os.path.join(path, obj)
            obj_name_without_extension, extension = os.path.splitext(obj)
            obj_name = getObjName(path_obj)
            
            Indexing[obj_name] = [obj_name_without_extension, None, None, None] 
            

print("Indexing shape test")
path = path_shape_test
for obj in os.listdir(path):
        if obj.endswith(extensions_3d):
            path_obj = os.path.join(path, obj)
            obj_name_without_extension, extension = os.path.splitext(obj)

            obj_name = getObjName(path_obj)
            if obj_name in Indexing:
                Indexing[obj_name] = [Indexing[obj_name][0], obj_name_without_extension, Indexing[obj_name][2], Indexing[obj_name][3]]  # modifica si ya existe
            else:
                Indexing[obj_name] = [None, obj_name_without_extension, None, None]  # crea si no existe

print("Indexing culture train")
path = path_culture_train
for obj in os.listdir(path):
        if obj.endswith(extensions_3d):
            path_obj = os.path.join(path, obj)
            obj_name_without_extension, extension = os.path.splitext(obj)

            obj_name = getObjName(path_obj)
            if obj_name in Indexing:
                Indexing[obj_name] = [Indexing[obj_name][0], Indexing[obj_name][1], obj_name_without_extension, Indexing[obj_name][3]]  # modifica si ya existe
            else:
                Indexing[obj_name] = [None, None, obj_name_without_extension, None]  # crea si no existe

print("Indexing culture test")
path = path_culture_test
for obj in os.listdir(path):
        if obj.endswith(extensions_3d):
            path_obj = os.path.join(path, obj)
            obj_name_without_extension, extension = os.path.splitext(obj)

            obj_name = getObjName(path_obj)
            if obj_name in Indexing:
                Indexing[obj_name] = [Indexing[obj_name][0], Indexing[obj_name][1], Indexing[obj_name][2], obj_name_without_extension]  # modifica si ya existe
            else:
                Indexing[obj_name] = [None, None, None, obj_name_without_extension]  # crea si no existe

import json

Indexing = dict(sorted(Indexing.items()))
with open("Indexing.json", "w") as f:
    json.dump(Indexing, f)