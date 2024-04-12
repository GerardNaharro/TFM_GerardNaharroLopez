import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops
import cv2

'''def eliminar_regiones_pequenas(imagen_binaria, umbral_area = 6000):
    # Etiquetar las regiones conectadas en la imagen
    etiquetas, num_etiquetas = label(imagen_binaria)

    # Obtener las propiedades de las regiones etiquetadas
    propiedades = regionprops(etiquetas)

    # Crear una máscara para almacenar las regiones que deseamos mantener
    mascara = np.zeros_like(imagen_binaria)

    # Filtrar las regiones cuyo tamaño es mayor o igual que el umbral especificado
    for region in propiedades:
        if region.area >= umbral_area:
            mascara[etiquetas == region.label] = 1

    # Aplicar la máscara a la imagen binaria original para eliminar las regiones pequeñas
    imagen_filtrada = imagen_binaria * mascara

    return imagen_filtrada

# Ejemplo de uso
imagen_binaria = cv2.imread("imagen.jpg")

# Definir el umbral de área
umbral_area = 10000

# Eliminar regiones pequeñas
imagen_filtrada = eliminar_regiones_pequenas(imagen_binaria, umbral_area)

print("Imagen binaria original:")
cv2.imshow("original",imagen_binaria)

print("\nImagen binaria filtrada:")
cv2.imshow("fitrada",imagen_filtrada)


imagen_filtrada2 = cv2.dilate(imagen_filtrada, np.ones((1, 1), np.uint8), iterations=2)
cv2.imshow("fitrada2",imagen_filtrada2)
cv2.waitKey(0)'''

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import pandas as pd

# Directorio donde se encuentran los archivos CSV
directorio = r'C:\Users\gerar\PycharmProjects\TFM\parametersAcotados_csv'

r'''
# Cargar los datos desde el archivo CSV
data = pd.read_csv(r'C:\Users\gerar\PycharmProjects\TFM\parameters_csv/ASMvsMCY_newGaussianParameters.csv', sep=";")
print(data)
# Obtener las columnas
x = data["param1"]
y = data["param2"]
z1 = data["good"]  # Frames bien etiquetados
z2 = data["bad"]  # Frames mal etiquetados

# Crear la figura y el eje
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
ax.plot_trisurf(x, y, z1, cmap='viridis', edgecolor='none')

# Añadir etiquetas
ax.set_xlabel('Parámetro 1')
ax.set_ylabel('Parámetro 2')
ax.set_zlabel('Frames Bien Etiquetados')

# Mostrar el gráfico
plt.show()
'''

# Obtener la lista de archivos CSV en el directorio
archivos_csv = glob.glob(os.path.join(directorio, '*.csv'))

# Crear un diccionario para almacenar los datos
datos = {}
data_sum = pd.DataFrame()

# Iterar sobre cada archivo CSV y cargarlo en un DataFrame
for ind,archivo in enumerate(archivos_csv):
    # Obtener el nombre del archivo sin la extensión
    nombre_archivo = os.path.splitext(os.path.basename(archivo))[0]
    # Cargar el archivo CSV en un DataFrame
    datos[ind] = pd.read_csv(archivo, sep=";")

data_sum['param1'] = datos[0]['param1']
data_sum['param2'] = datos[0]['param2']

for i in range(len(datos)):
    datos[i]['param1'] = datos[0]['param1']
    datos[i]['param2'] = datos[0]['param2']
    datos[i].to_csv(directorio + '/procesados_' + str(i) +'.csv', index= False)


