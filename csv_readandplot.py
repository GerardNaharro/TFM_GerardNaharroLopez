import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import pandas as pd

# Directorio donde se encuentran los archivos CSV
directorio = r'C:\Users\gerar\PycharmProjects\TFM\parametersAcotados_csv\procesados'

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
data_sum['good'] = 0
data_sum['bad'] = 0

for i in range(len(datos)):
    data_sum['good'] += datos[i]['good']
    data_sum['bad'] += datos[i]['bad']

data_sum['good'] = data_sum['good'] / len(datos)
data_sum['bad'] = data_sum['bad'] / len(datos)
print(data_sum)

filas_maximas = data_sum.loc[data_sum['good'] == data_sum['good'].max()]

print("Filas con el valor más alto en la columna 'good':")
print(filas_maximas)

# Obtener los índices de las filas maximas
indices_maximos = filas_maximas.index
print(indices_maximos)
for i in range(len(datos)):
    print(datos[i].loc[indices_maximos])

# Obtener las columnas
x = data_sum["param1"]
y = data_sum["param2"]
z1 = data_sum["good"]  # Frames bien etiquetados
z2 = data_sum["bad"]  # Frames mal etiquetados

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
