import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import product
import pandas as pd
import seaborn as sns

def fnAprender(MA, dato):
    MA += 2 * dato - 1
    return MA

def fnRecuperar(MA, x):
    return np.argmax(MA @ np.transpose(x))

# Función para extraer características de una imagen
def extract_features(image_path, nbits):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    imb = np.array(img) > 128

    roi_y_range = range(5, 24)
    roi_x_range = range(6, 22)

    all_points = list(product(roi_y_range, roi_x_range))
    np.random.seed(82)
    selected_points = np.random.choice(len(all_points), nbits, replace=False)
    xx = [all_points[i][0] for i in selected_points]
    yy = [all_points[i][1] for i in selected_points]

    VectorCod = np.zeros((nbits))
    for i in range(nbits):
        VectorCod[i] = imb[xx[i], yy[i]]

    return VectorCod, xx, yy, img

# Ruta de las imágenes de entrenamiento y prueba
train_path = r'Imagenes_entrenamiento'
test_path = r'Imagenes_test'

# Fase de aprendizaje
nc = 10  # Número de clases (0-9)
nbits = 150  # Número de bits/características

MA = np.zeros((nc, nbits))
clases = []

# Variables para almacenar la última imagen y sus puntos de muestreo
ultima_img = None
ultima_xx = None
ultima_yy = None

for archivo in os.listdir(train_path):
    match = re.match(r'training_image\[(\d+)\]_(\d+)\.jpg', archivo)
    if match:
        clase = int(match.group(1))
        if clase not in clases:
            clases.append(clase)
        ruta_imagen = os.path.join(train_path, archivo)
        features, xx, yy, img = extract_features(ruta_imagen, nbits)
        MA[clase, :] = fnAprender(MA[clase, :], features)
        # Actualizar la última imagen y sus puntos de muestreo
        ultima_img = img
        ultima_xx = xx
        ultima_yy = yy

# Evaluación
confusion_matrix = np.zeros((nc, nc), dtype=int)

for archivo in os.listdir(test_path):
    match = re.match(r'test_image\[(\d+)\]_(\d+)\.jpg', archivo)
    if match:
        clase_verdadera = int(match.group(1))
        ruta_imagen = os.path.join(test_path, archivo)
        test_features, _, _, _ = extract_features(ruta_imagen, nbits)
        prediccion = fnRecuperar(MA, test_features)
        confusion_matrix[clase_verdadera, prediccion] += 1

print("Matriz de Aprendizaje (MA):")
print(MA)
print("Matriz de Confusión:")
print(confusion_matrix)

# Calcular la precisión
correct_predictions = np.trace(confusion_matrix)
total_predictions = np.sum(confusion_matrix)
precision = correct_predictions / total_predictions
print("Precisión:", precision * 100)

# Guardar la matriz de aprendizaje en un archivo Excel
df = pd.DataFrame(MA)
df.to_excel('matriz_aprendizaje.xlsx', index=False)
print("Matriz de aprendizaje guardada en 'matriz_aprendizaje.xlsx'")

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.title(f'Matriz de Confusión {precision*100:.2f} % de Precisión')
plt.show()

# Mostrar la última imagen con sus puntos de muestreo aleatorio
plt.figure(figsize=(6, 6))
plt.imshow(ultima_img, cmap='gray')
plt.scatter(ultima_yy, ultima_xx, color='red', s=10)  # Intercambiamos xx y yy para que los puntos se alineen correctamente
plt.title('Última Imagen con Puntos de Muestreo Aleatorio')
plt.show()
