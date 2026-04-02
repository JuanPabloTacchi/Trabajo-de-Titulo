
import os
import shutil

#Final step, merge the cloud, images and text into the final dataset folder


def createTriplet(ruta_imagenes, ruta_textos, ruta_3d, ruta_out, nombre_objeto):    
    
   
    nombre_out_pc = os.path.join(ruta_out, nombre_objeto)
    nombre_out_pc = nombre_out_pc + ".ply"

    shutil.copy(ruta_3d, nombre_out_pc)
    
    for img in os.listdir(ruta_imagenes):
    # Verificar si el archivo tiene una extensión de imagen
        
        # Obtener la ruta completa del archivo
        nombre_img , extension = os.path.splitext(img)
        ruta_imagen = os.path.join(ruta_imagenes, img)

        dir_img_out = os.path.join(ruta_out, nombre_img)
        os.makedirs(dir_img_out, exist_ok=True)
        shutil.copy(ruta_imagen, dir_img_out)
        
        nombre_imagen, extension = os.path.splitext(img)
        print(nombre_imagen)
        print(ruta_textos)
        ruta_textos_de_imagen = os.path.join(ruta_textos, nombre_imagen)
        print(ruta_textos_de_imagen)
        for txt in os.listdir(ruta_textos_de_imagen):
            nombre_txt , extension = os.path.splitext(txt)
            ruta_texto_de_imagen = os.path.join(ruta_textos_de_imagen, txt)

            dir_txt_out = os.path.join(dir_img_out, nombre_txt)
            os.makedirs(dir_txt_out, exist_ok=True)
            shutil.copy(ruta_texto_de_imagen, dir_txt_out)

    

def createDataset(ruta_imagenes, ruta_textos, ruta_pc, ruta_out):
    
    for pc in os.listdir(ruta_pc):
        nombre_objeto, extension = os.path.splitext(pc)
        ruta_archivo_pc = os.path.join(ruta_pc, pc)
        ruta_archivo_texto = os.path.join(ruta_textos, nombre_objeto)
        ruta_archivo_imagen = os.path.join(ruta_imagenes, nombre_objeto)
        ruta_archivo_out = os.path.join(ruta_out, nombre_objeto)
        os.makedirs(ruta_archivo_out, exist_ok=True)
        createTriplet(ruta_archivo_imagen, ruta_archivo_texto, ruta_archivo_pc, ruta_archivo_out, nombre_objeto)          
    print("termino")

#add your own paths here
ruta_pc = ".../herital/both/cloud/train"
ruta_img = ".../herital/both/images/train"
ruta_txt = ".../herital/both/texts/train"
ruta_out = ".../herital/herital/final_dataset/train"
#repeat for test and val
createDataset(ruta_img, ruta_txt, ruta_pc, ruta_out)