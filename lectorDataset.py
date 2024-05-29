import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt
import os

def read_images(images_path, labels_path, save_path, max_per_category=None):
    # Verificar si los directorios ya existen
    if os.path.exists(save_path):
        print(f"El directorio '{save_path}' ya existe. No se realizarán cambios.")
        return

    # Leer las etiquetas de los identificadores
    with open(labels_path, 'rb') as labels_file:
        magic, size = struct.unpack(">II", labels_file.read(8))
        labels_data = array("B", labels_file.read())

    # Crear la carpeta si no existe
    os.makedirs(save_path)

    # Contadores por número
    counters = [0] * 10  # Se asume que hay 10 números diferentes en las etiquetas

    # Leer las imágenes y guardarlas con su identificador y número correspondiente
    with open(images_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        image_data = array("B", file.read())

        for i in range(size):
            # Verificar si se ha alcanzado la cantidad máxima de imágenes por categoría
            if max_per_category is not None and all(counters[label] >= max_per_category for label in range(10)):
                print(f"Se ha alcanzado el límite de {max_per_category} imágenes por categoría. No se guardarán más.")
                break

            image = np.array(image_data[i * rows * cols: (i + 1) * rows * cols])
            image = image.reshape((rows, cols))

            # Obtener el identificador y número correspondiente de la imagen
            image_id = i
            image_label = labels_data[i]

            # Verificar si se ha alcanzado el límite de imágenes para esta categoría
            if max_per_category is not None and counters[image_label] >= max_per_category:
                continue

            # Obtener el contador para este número y actualizarlo
            counter = counters[image_label]
            counters[image_label] += 1

            # Guardar la imagen con su identificador y número en el nombre del archivo
            if "train" in images_path:
                plt.imsave(
                    os.path.join(save_path, f"training_image[{image_label}]_{counter}.jpg"),
                    image,
                    cmap='gray'
                )
            elif "t10k" in images_path:
                plt.imsave(
                    os.path.join(save_path, f"test_image[{image_label}]_{counter}.jpg"),
                    image,
                    cmap='gray'
                )

    print(f"¡Proceso de carga de imágenes en '{save_path}' finalizado con éxito!")

train_images_path = r'archive\train-images-idx3-ubyte\train-images-idx3-ubyte'
train_labels_path = r'archive\train-labels-idx1-ubyte\train-labels-idx1-ubyte'
test_images_path = r'archive\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte'
test_labels_path = r'archive\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte'

test_save_path = r"Imagenes_test"
train_save_path = r"Imagenes_entrenamiento"

# Aquí defines la cantidad máxima de imágenes por categoría que deseas guardar
#read_images(train_images_path, train_labels_path, train_save_path, max_per_category=2000)
#read_images(test_images_path, test_labels_path, test_save_path, max_per_category=860)
read_images(train_images_path, train_labels_path, train_save_path, max_per_category=2000)
read_images(test_images_path, test_labels_path, test_save_path, max_per_category=860)