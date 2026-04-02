import json
from collections import Counter
from datasetCreationUtils import getObjName
import os
import open3d as o3d
from mesh_utils import sample_mesh_with_texture, save_point_cloud_with_colors
import math

#Transform from the original dataset mesh format into a point cloud format

def mesh2cloud(in_path, out_path):
    points, colors = sample_mesh_with_texture(in_path, num_points=100000)
    # Guardar la nube de puntos
    save_point_cloud_with_colors(points, colors, out_path)


#paths
#Add your own paths here
path_shape_train_in = ".../herital/shape/train"
path_shape_test_in = ".../herital/shape/test"
path_dataset = ".../herital/both/cloud"

#dataset percentage
val_percentage = 0.1
test_percentage = 0.2

#Changes mesh .obj to cloud .ply
#Also transform indexes 
Indexing = None
with open("Indexing.json", "r") as f:
    Indexing = json.load(f)

Obj_Index = None
with open("obj_indexer.json", "r") as f:
    Obj_Index = json.load(f)

print("Starting")
#gets dict with label1_label2: n of label1_label2 in the dataset
obj_by_types = Counter(Obj_Index.values())

#dict with key: obj_name, value: train/test
dict_with_dir_path_by_index = {}

print("Separating the data")
#Separates objects into train/test
from collections import defaultdict
import random

grouped = defaultdict(list)

for k, v in Obj_Index.items():
    grouped[v].append(k)

#For each label, it takes a representative amount of elements
labels = []
for label in dict(grouped):
    arr = grouped[label]
    arr = random.shuffle(arr)
    n = len(arr)
    test_arr = arr[0:math.ceil(n*test_percentage)]
    total_train_arr = arr[math.ceil(n*test_percentage) : n]
    
    n2 = len(total_train_arr)
    val_arr = total_train_arr[0 : math.ceil(n2*val_percentage)]
    train_arr = total_train_arr[math.ceil(n2*val_percentage) : n2]
    for i in range(len(test_arr)):
        dict_with_dir_path_by_index[test_arr[i]] = "test"
    for i in range(len(val_arr)):
        dict_with_dir_path_by_index[val_arr[i]] = "val"
    for i in range(len(train_arr)):
        dict_with_dir_path_by_index[train_arr[i]] = "train"
print(dict_with_dir_path_by_index)
    
    
extensions_3d = ('.obj')

print("Saving clouds")
#Saves the object in .ply form
for obj in os.listdir(path_shape_train_in):
        if obj.endswith(extensions_3d):
            path_obj = os.path.join(path_shape_train_in, obj)
            obj_name_without_extension, extension = os.path.splitext(obj)
            obj_index = getObjName(path_obj)
            ruta_ply = os.path.join(path_dataset,dict_with_dir_path_by_index[obj_index])
            ruta_ply = os.path.join(ruta_ply,obj_index) 
            ruta_ply = ruta_ply + ".ply" 
            mesh2cloud(path_obj,ruta_ply)


for obj in os.listdir(path_shape_test_in):
        if obj.endswith(extensions_3d):
            path_obj = os.path.join(path_shape_test_in, obj)
            obj_name_without_extension, extension = os.path.splitext(obj)
            obj_index = getObjName(path_obj)
            ruta_ply = os.path.join(path_dataset,dict_with_dir_path_by_index[obj_index])
            ruta_ply = os.path.join(ruta_ply,obj_index) 
            ruta_ply = ruta_ply + ".ply" 
            mesh2cloud(path_obj,ruta_ply)

            
print("Termino")
