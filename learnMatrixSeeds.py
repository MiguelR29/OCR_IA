import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import product
import pandas as pd

def fnAprender(MA, dato):
    MA += 2 * dato - 1
    return MA

def fnRecuperar(MA, x):
    return np.argmax(MA @ np.transpose(x))

# Función para extraer características de una imagen con una semilla específica
def extract_features_with_seed(image_path, nbits, seed):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))  # Asegúrate de que todas las imágenes tengan el mismo tamaño
    imb = np.array(img) > 128

    # Muestreo aleatorio
    #all_points = list(product(range(7, 21), range(4, 23)))
    #all_points = list(product(range(5, 25), range(6, 23)))
    #all_points = list(product(range(1, 28), range(1, 28)))
    all_points = list(product(range(5, 23), range(5, 25)))    
    

    np.random.seed(seed)
    selected_points = np.random.choice(len(all_points), nbits, replace=False)
    xx = [all_points[i][0] for i in selected_points]
    yy = [all_points[i][1] for i in selected_points]

    VectorCod = np.zeros((nbits))
    for i in range(nbits):
        VectorCod[i] = imb[xx[i], yy[i]]

    return VectorCod

# Función para encontrar una semilla común que equilibre los bits en las características de todas las imágenes
def find_common_seed(image_paths, nbits, seed_start=0):
    seed = seed_start
    while True:
        print('Semilla en proceso', seed)
        all_balanced = True
        for image_path in image_paths:
            print('Imagen en proceso', image_path)
            features = extract_features_with_seed(image_path, nbits, seed)
            print('Características', features)
            if np.sum(features) < nbits / 2:                
                print('Características no balanceadas, sumatoria: ', np.sum(features))
                all_balanced = False
                break
        if all_balanced:
            return seed
        seed += 1

# Ruta de las imágenes de entrenamiento y prueba
train_path = r'C:\Universidad\Sexto_Semestre\IA\Proyecto_final\Imagenes_entrenamiento'
test_path = r'C:\Universidad\Sexto_Semestre\IA\Proyecto_final\Imagenes_test'

# Fase de aprendizaje
nc = 10  # Número de clases (0-9)
nbits = 110  # Número de bits/características

MA = np.zeros((nc, nbits))
clases = []
train_image_paths = []

# Recopilar rutas de imágenes de entrenamiento
for archivo in os.listdir(train_path):
    match = re.match(r'training_image\[(\d+)\]_(\d+)\.jpg', archivo)
    if match:
        train_image_paths.append(os.path.join(train_path, archivo))

# Encontrar una semilla común que equilibre los bits para todas las imágenes de entrenamiento
common_seed = find_common_seed(train_image_paths, nbits)
print(f'Common Seed Found: {common_seed}')

# Utilizar la semilla común para extraer características y entrenar el modelo
for archivo in os.listdir(train_path):
    match = re.match(r'training_image\[(\d+)\]_(\d+)\.jpg', archivo)
    if match:
        clase = int(match.group(1))
        if clase not in clases:
            clases.append(clase)
        ruta_imagen = os.path.join(train_path, archivo)
        features = extract_features_with_seed(ruta_imagen, nbits, common_seed)
        print(f'Features: {features}')
        MA[clase, :] = fnAprender(MA[clase, :], features)

# Evaluación
# confusion_matrix = np.zeros((nc, nc), dtype=int)

# for archivo in os.listdir(test_path):
#     match = re.match(r'test_image\[(\d+)\]_(\d+)\.jpg', archivo)
#     if match:
#         clase_verdadera = int(match.group(1))
#         ruta_imagen = os.path.join(test_path, archivo)
#         test_features = extract_features_with_seed(ruta_imagen, nbits, common_seed)
#         prediccion = fnRecuperar(MA, test_features)
#         confusion_matrix[clase_verdadera, prediccion] += 1

# print("Matriz de Aprendizaje (MA):")
# print(MA)
# print("Matriz de Confusión:")
# print(confusion_matrix)

# Calcular la precisión
# correct_predictions = np.trace(confusion_matrix)
# total_predictions = np.sum(confusion_matrix)
# precision = correct_predictions / total_predictions
# print("Precisión:", precision*100)

# Guardar la matriz de aprendizaje en un archivo Excel
# df = pd.DataFrame(MA)
# df.to_excel('matriz_aprendizaje.xlsx', index=False)
# print("Matriz de aprendizaje guardada en 'matriz_aprendizaje.xlsx'")

# Visualizar la matriz de confusión
# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 8))
# sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicción')
# plt.ylabel('Verdadero')
# plt.title(f'Matriz de Confusión {precision*100:.2f} % de Precisión')
# plt.show()
