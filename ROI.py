import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ruta de las imágenes de entrenamiento
train_path = r'C:\Universidad\Sexto_Semestre\IA\Proyecto_final\Imagenes_entrenamiento'

# Obtener la lista de archivos de imágenes en el directorio de entrenamiento
image_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.jpg')]

# Inicializar una matriz acumulativa
accumulated_image = None

for image_file in image_files:
    # Cargar la imagen y convertirla a escala de grises
    img = Image.open(image_file).convert('L')
    img_array = np.array(img)
    
    # Inicializar la imagen acumulativa la primera vez
    if accumulated_image is None:
        accumulated_image = np.zeros_like(img_array, dtype=np.float64)
    
    # Acumular la imagen
    accumulated_image += img_array

# Normalizar la imagen acumulada para visualización
accumulated_image /= len(image_files)

print(f"Se han acumulado {len(image_files)} imágenes de entrenamiento.")
# Mostrar la imagen acumulada
plt.imshow(accumulated_image, cmap='hot')
plt.colorbar()
plt.title('Superposición de Imágenes de Entrenamiento')
plt.show()