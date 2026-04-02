import os
import subprocess
import json

#Render images for each point cloud element on the dataset


def render_object(in_path, out_path):
    blender_command = [
        "blender", 
        "--background", 
        "--python", "render_blender.py", 
        "--", 
        "--output_folder", out_path, 
        in_path,
        "--scale", "0.006",
        "--views", "8"
    ]
    result = subprocess.run(blender_command, capture_output=True, text=True, check=True)
    print(result)
    

def render_dataset(in_path, out_path):
    extensiones_3d = ('.obj')

    # Recorrer cada archivo en el directorio
    #for archivo in os.listdir(ruta_imagenes):
        #ruta_archivo_imagen = os.path.join(ruta_imagenes, archivo)
    for obj in os.listdir(in_path):
    # Verificar si el archivo tiene una extensión de imagen
        if obj.endswith(extensiones_3d):
            ruta_obj = os.path.join(in_path, obj)
            nombre_obj_sin_extension, extension = os.path.splitext(obj)
            ruta_imagenes = os.path.join(out_path, nombre_obj_sin_extension)
            render_object(ruta_obj,ruta_imagenes)


Indexing = None
with open("Indexing.json", "r") as f:
    Indexing = json.load(f)
#add your own paths here
path_pc_train = ".../herital/both/cloud/train"
path_pc_test = ".../herital/both/cloud/test"
path_pc_val = ".../herital/both/cloud/val"
path_img_train = ".../herital/both/images/train"
path_img_test = ".../herital/both/images/test"
path_img_val = ".../herital/both/images/val"
#original mesh files
path_raw_pc_train = ".../herital/train"
path_raw_pc_test = ".../herital/test"


print("Starting")
print("Train")
extensions_3d = (".ply")
path_in = path_pc_train
path_out = path_img_train
for obj in os.listdir(path_in):
        if obj.endswith(extensions_3d):
            path_obj = os.path.join(path_in, obj)
            obj_name_without_extension, extension = os.path.splitext(obj)
            obj_index = obj_name_without_extension
            obj_old_index_list = Indexing[obj_index]
            if obj_old_index_list[0] != None:
                path_mesh_obj = os.path.join(path_raw_pc_train, obj_old_index_list[0] + ".obj")
            else:
                path_mesh_obj = os.path.join(path_raw_pc_test, obj_old_index_list[1] + ".obj")
            path_img_out = os.path.join(path_out, obj_name_without_extension)
            print(path_img_out)
            render_object(path_mesh_obj, path_img_out)



print("Test")
extensions_3d = (".ply")
path_in = path_pc_test
path_out = path_img_test
for obj in os.listdir(path_in):
        if obj.endswith(extensions_3d):
            path_obj = os.path.join(path_in, obj)
            obj_name_without_extension, extension = os.path.splitext(obj)
            obj_index = obj_name_without_extension
            obj_old_index_list = Indexing[obj_index]
            if obj_old_index_list[0] != None:
                path_mesh_obj = os.path.join(path_raw_pc_train, obj_old_index_list[0] + ".obj")
            else:
                path_mesh_obj = os.path.join(path_raw_pc_test, obj_old_index_list[1] + ".obj")
            path_img_out = os.path.join(path_out, obj_name_without_extension)
            print(path_img_out)
            render_object(path_mesh_obj, path_img_out)



print("Val")
extensions_3d = (".ply")
path_in = path_pc_val
path_out = path_img_val
for obj in os.listdir(path_in):
        if obj.endswith(extensions_3d):
            path_obj = os.path.join(path_in, obj)
            obj_name_without_extension, extension = os.path.splitext(obj)
            obj_index = obj_name_without_extension
            obj_old_index_list = Indexing[obj_index]
            if obj_old_index_list[0] != None:
                path_mesh_obj = os.path.join(path_raw_pc_train, obj_old_index_list[0] + ".obj")
            else:
                path_mesh_obj = os.path.join(path_raw_pc_test, obj_old_index_list[1] + ".obj")
            path_img_out = os.path.join(path_out, obj_name_without_extension)
            print(path_img_out)
            render_object(path_mesh_obj, path_img_out)

print("Termino")


    
   
