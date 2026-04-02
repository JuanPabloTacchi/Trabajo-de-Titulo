from PIL import Image
import os
import shutil

def getObjName(path_obj):
    with open(path_obj, 'r') as text:
        raw_text = text.read()
        obj_name = raw_text.split("Object")[1].split("\n")[0].strip(" ").split(".")[0]
    return obj_name

def countObjects(ruta):
    i = 0
    for dir in os.listdir(ruta):
        i = i + 1
    return i

def getDirWithIndex(i):
    j = 0
    for dir in os.listdir(ruta):
        j = j + 1
        if i == j:
            return dir
    return "Ended"

def copyUntilIndex(dir_in, dir_out, i):
    j = 0
    for dir in os.listdir(dir_in):
        dir_in_object = os.path.join(dir_in, dir)
        dir_out_object = os.path.join(dir_out, dir)
        shutil.copy(dir_in_object, dir_out_object)
        j = j + 1
        if i == j:
            return dir
    return "Ended"
