import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
from itertools import product

def fnRecuperar(MA, x):
    return np.argmax(MA @ np.transpose(x))

def extract_features(image_path, nbits):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    imb = np.array(img) > 128

    #all_points = list(product(range(7, 21), range(4, 23)))
    #all_points = list(product(range(8, 21), range(3, 27)))
    #all_points = list(product(range(5, 25), range(6, 23)))
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

    return VectorCod, img, xx, yy

def cargar_imagen():
    file_path = filedialog.askopenfilename()
    if file_path:
        features, img, xx, yy = extract_features(file_path, nbits)

        # Mostrar la imagen original
        img_original = ImageTk.PhotoImage(img.resize((200, 200)))
        panel_original.config(image=img_original)
        panel_original.image = img_original

        # Mostrar la imagen con puntos de muestreo
        img_with_points = img.copy().convert('RGB')
        draw = ImageDraw.Draw(img_with_points)
        for x, y in zip(xx, yy):
            draw.ellipse((y-1, x-1, y+1, x+1), fill='red')
        img_with_points = ImageTk.PhotoImage(img_with_points.resize((200, 200)))
        panel_points.config(image=img_with_points)
        panel_points.image = img_with_points

        # Evaluar la imagen y mostrar el resultado
        resultado = evaluar_imagen(features)
        resultado_label.config(text=f'Predicción: {resultado}')

def evaluar_imagen(test_features):
    prediccion = fnRecuperar(MA, test_features)
    return prediccion

# Cargar la matriz de aprendizaje desde el archivo Excel
df = pd.read_excel('matriz_aprendizaje.xlsx')
MA = df.to_numpy()

nc = MA.shape[0]    
nbits = 150  # Número de bits/características

# Crear la interfaz gráfica con Tkinter
root = tk.Tk()
root.title('Clasificación de Imágenes')

# Configuración de colores
bg_color = '#245657'
btn_color = '#2E8B57'
text_color = '#FFFFFF'

# Configurar la geometría de la ventana para que aparezca en el centro
window_width = 800
window_height = 600

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
root.configure(bg=bg_color)

# Crear un botón estilizado
btn = tk.Button(root, text='Cargar Imagen de Prueba', command=cargar_imagen, bg=btn_color, fg=text_color,
                font=('Helvetica', 16, 'bold'), bd=0, highlightthickness=0, relief='flat', padx=20, pady=10)
btn.pack(pady=20)

# Usar un marco para contener el panel de la imagen original y con puntos
frame = tk.Frame(root, bg=bg_color)
frame.pack(pady=20)

panel_original = tk.Label(frame, bg=bg_color)
panel_original.grid(row=0, column=0, padx=10)

panel_points = tk.Label(frame, bg=bg_color)
panel_points.grid(row=0, column=1, padx=10)

resultado_label = tk.Label(root, text='Predicción:', bg=bg_color, fg=text_color, font=('Helvetica', 14))
resultado_label.pack(pady=10)

root.mainloop()
