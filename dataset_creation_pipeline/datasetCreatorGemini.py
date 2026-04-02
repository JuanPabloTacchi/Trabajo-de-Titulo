from google import genai
from google.genai import types
from PIL import Image
from datasetCreationUtils import countObjects
import os
import json
import time

#script to generate the text features of the images, it can count if there was an error (i had a lot with the amount of
# data sent, every now and then it crashed from the gemini side), so it will start over from the last file it was generating text

client = genai.Client(api_key="")

Indexes = None
with open("Indexing.json", "r") as f:
    Indexes = json.load(f)

Shapes_train = None
with open("shapes_train.json", "r") as f:
    Shapes_train = json.load(f)

Shapes_test = None
with open("shapes_test.json", "r") as f:
    Shapes_test = json.load(f)

Culture_train = None
with open("cultures_train.json", "r") as f:
    Culture_train = json.load(f)

Culture_test = None
with open("cultures_test.json", "r") as f:
    Culture_test = json.load(f)

def withCulturePrompts(shape, culture):
    prompts = []
    prompt = """You are an expert archaeologist with extensive knowledge in analyzing ancient artifacts.
                            Please describe the following object based on its shape, color, texture, and patterns.
                            Provide a detailed description of the shape of the object. Is it geometric, organic, irregular, etc. Mention any notable features that define its structure. 
                            Take into account that it is a """ +shape+ """.
                            Describe the primary and secondary colors of the object. Are there any color variations or fading.
                            Explain the texture of the surface. Is it smooth, rough, bumpy, worn down, etc.
                            If there are any pattern, designs or markings, describe them. Do not say nothing more than an accurrate description of the object, dont give assumptions.
                            Describe the object without headings or specific breaks. Integrate the information naturally. . Take into account and mention that it is from the culture """ + culture + """.
                            Start the description with: a """ +shape+ """ ... """
    prompts.append(prompt)

                #muy meh
    prompt = """You are an expert archaeologist with extensive knowledge in analyzing ancient artifacts. Please describe the following object based on its shape, color, texture, and patterns. 
                        Provide as much detail as possible for each of these aspects. Do not say nothing more than an accurrate description of the object, dont give assumptions.
                        dont give an introduction for the description for example "Here is a description of the bowl based on the image:". Separate the sections in headers but only in text for example dont use **.
                        Dont say things like "Based on the image", just give the description,
                         also dont say things like "possibly" only say things that are certain. Consider that it is a """ + shape + """.
                        Take into account and mention that it is from the culture """ + culture + """. You have to mention the culture in a separate header: Culture"""
    prompts.append(prompt)

    prompt = """You are an expert archaeologist, using the knowledge you have on patterns in pottery from Peru, describe in detail the pattern in the object. Try to associate the previous knowledge to the patterns that are visible only if it is very certain of it.
                  Do not say nothing more than an description of the object, dont give assumptions. Consider the object is a """ +shape+ """. 
                  Take into account and mention that it is from the culture """ + culture + """. Start the description with: a """ +shape+""" from the """ + culture + """ culture ..."""
    prompts.append(prompt)

                

            
    prompt = "You are an expert archaeologist, describe the " + shape+ " in a very general way. Do not say nothing more than an description of the object, dont give assumptions. . Take into account and mention that it is from the culture " + culture + " Start the description with: a " + shape+" from the "+ culture+" culture ..." 
    prompts.append(prompt)
                
    prompt = """Following the next example of structure for cultural heritage objects describe the object using the knowledge in the area, dont give assumptions, consider the object is a """ + shape +""". . Take into account and mention that it is from the culture """ + culture + """
                         dont give an introduction for the description for example "Here is a description of the bowl based on the image:". Separate the sections in headers but only in text for example dont use **.
                        Preferred Name: Bowl
                        Physical Description: Restricted container with a globular body, rounded lip, and curved base. Smoothed, polished, and slipped surface, dark red on the inside and outside.
                        Material and Technique: Ceramic, modeling, smoothing, slipping.
                        Condition: Good
                        Culture: Chancay
                        """                
    prompts.append(prompt)
    return prompts

def withoutCulturePrompts(shape):
    prompts = []

                # Procesar la imagen y generar una descripción
    prompt = """You are an expert archaeologist with extensive knowledge in analyzing ancient artifacts.
                Please describe the following object based on its shape, color, texture, and patterns.
                Provide a detailed description of the shape of the object. Is it geometric, organic, irregular, etc. Mention any notable features that define its structure. 
                Take into account that it is a """ +shape+ """.
                Describe the primary and secondary colors of the object. Are there any color variations or fading.
                Explain the texture of the surface. Is it smooth, rough, bumpy, worn down, etc.
                If there are any pattern, designs or markings, describe them. Do not say nothing more than an accurrate description of the object, dont give assumptions.
                Describe the object without headings or specific breaks. Integrate the information naturally. Start the description with: a """ +shape+ """ ... """
    prompts.append(prompt)

                #muy meh
    prompt = """You are an expert archaeologist with extensive knowledge in analyzing ancient artifacts. Please describe the following object based on its shape, color, texture, and patterns. 
                Provide as much detail as possible for each of these aspects. Do not say nothing more than an accurrate description of the object, dont give assumptions.
                dont give an introduction for the description for example "Here is a description of the bowl based on the image:". Separate the sections in headers but only in text for example dont use **.
                Dont say things like "Based on the image", just give the description, also dont say things like "possibly" only say things that are certain. Consider that it is a """ + shape + """. """
    prompts.append(prompt)

    prompt = """You are an expert archaeologist, using the knowledge you have on patterns in pottery from Peru, describe in detail the pattern in the object. Try to associate the previous knowledge to the patterns that are visible only if it is very certain of it.
                Do not say nothing more than an description of the object, dont give assumptions. Consider the object is a """ +shape+ """. Start the description with: a """ +shape+""" ... """ 
    prompts.append(prompt)

                

            
    prompt = "You are an expert archaeologist, describe the " + shape+ " in a very general way. Do not say nothing more than an description of the object, dont give assumptions. Start the description with: a " + shape+" ..." 
    prompts.append(prompt)
                
    prompt = """Following the next example of structure for cultural heritage objects describe the object using the knowledge in the area, dont give assumptions, consider the object is a """ + shape +"""
                 dont give an introduction for the description for example "Here is a description of the bowl based on the image:". Separate the sections in headers but only in text for example dont use **.
                Preferred Name: Bowl
                Physical Description: Restricted container with a globular body, rounded lip, and curved base. Smoothed, polished, and slipped surface, dark red on the inside and outside.
                Material and Technique: Ceramic, modeling, smoothing, slipping.
                Condition: Good
                """                
    prompts.append(prompt)

    return prompts

def main(ruta_textos, ruta_imagenes, i = 0):
    # Cargar el modelo y el procesador

    # Extensiones de archivo de imágenes que quieres recorrer
    extensiones_imagenes = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    j = 0
    # Recorrer cada archivo en el directorio
    for archivo in os.listdir(ruta_imagenes):
        if j < i:
            j = j + 1
            print("archivo " + archivo + " saltado")
            continue
        ruta_archivo_imagen = os.path.join(ruta_imagenes, archivo)

        for imagen in os.listdir(ruta_archivo_imagen):
        # Verificar si el archivo tiene una extensión de imagen
            if imagen.endswith(extensiones_imagenes):
                # Obtener la ruta completa del archivo
                ruta_imagen = os.path.join(ruta_archivo_imagen, imagen)
                # Cargar una imagen
                image = Image.open(ruta_imagen)
                nombre_imagen_sin_extension, extension = os.path.splitext(imagen)
                output_path = os.path.join(ruta_textos, archivo)
                os.makedirs(output_path, exist_ok=True)
                output_path = os.path.join(output_path, nombre_imagen_sin_extension)
                os.makedirs(output_path, exist_ok=True)
                
                prompts = []

                indexes = Indexes[nombre_imagen_sin_extension.split("_")[0]]

                if indexes[0] != None:
                    shape = Shapes_train[indexes[0]].lower()
                    print("Shape: " + shape)
                if indexes[1] != None:
                    shape = Shapes_test[indexes[1]].lower()
                    print("Shape: " + shape)
                if indexes[2] != None:
                    culture = Culture_train[indexes[2]].lower()
                    print("Culture: " + culture)
                    prompts = withCulturePrompts(shape,culture)
                elif indexes[3] != None:
                    culture = Culture_test[indexes[3]].lower()
                    print("Culture: " + culture)
                    prompts = withCulturePrompts(shape,culture)
                else:
                    prompts = withoutCulturePrompts(shape)

                
                ## Generar la descripción
                k = 0
                while k < len(prompts):
#               #default max_lenght = 20
                    try:
                        print(output_path + ": ")
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[image, prompts[k]])
                        print(response.text)
                        final_text = response.text
                        output_path_text = output_path + "/"+ str(k) +  ".txt"
                        with open(output_path_text, 'w') as file:
                            file.write(final_text)
                        k = k+1
                    
                    except Exception as e:
                        print("Ocurrió un error:", e)
                        print("El error ocurrio en el i=", j)
                        time.sleep(180)
                    
                
                k = 0
                while k < len(prompts):
                    try:
                        print(output_path + ": ")
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[image, prompts[k]],
                            config=types.GenerateContentConfig(
                            max_output_tokens=200,
                            temperature=0.1
                            ))
                        print(response.text)

                        final_text = response.text
                        output_path_text = output_path + "/"+  str(k+5) +  ".txt"
                        with open(output_path_text, 'w') as file:
                            file.write(final_text)
                        k = k+1
                    except Exception as e:
                        print("Ocurrió un error:", e)
                        print("El error ocurrio en el i=", j)
                        time.sleep(180)
            

                
    print("termino")


Train
#add your own paths here
#img path
rutatestimagenes = ".../herital/both/images/train"
#txt path
rutatesttextos = ".../herital/both/texts/train"
n_objectos_procesados = countObjects(rutatesttextos)
main(rutatesttextos, rutatestimagenes, n_objectos_procesados)

#Val
rutatestimagenes = ".../herital/both/images/val"
rutatesttextos = ".../herital/both/texts/val"
n_objectos_procesados = countObjects(rutatesttextos)
main(rutatesttextos, rutatestimagenes, n_objectos_procesados)

#Test
rutatestimagenes = ".../herital/both/images/test"
rutatesttextos = ".../herital/both/texts/test"
n_objectos_procesados = countObjects(rutatesttextos)
main(rutatesttextos, rutatestimagenes, n_objectos_procesados)