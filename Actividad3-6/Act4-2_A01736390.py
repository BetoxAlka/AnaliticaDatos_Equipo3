# Actividad 4.2: Clasificador Knn

# Importacion de librerias necesarias para trabajar
import pandas as pd # Importar Pandas
from pandas import DataFrame # Importar librerias necesarias para trabajar con Frames

def FrameFromCSV(filename): # Funcion que lee el archivo csv con los datos, recibe el nombre del archivo
    frame = pd.read_csv(filename) # Metodo para crear Frame a partir de archivo csv

    return frame # Regresa el Frame resultante

# Funcion para calcular las distancias de cada instancia del clasificador, 
def EuclidicKNN(datos,prueba): # Recibe el Frame con los datos y otro con los clasificadores
    colAName = datos.columns[0] # Nombre de la Primera Columna
    colBName = datos.columns[1] # Nombre de la Segunda Columna
    #colClass = datos.columns[2] # Nombre de la Columna con la Clase

    num_pruebas = prueba.shape[0] # Se obtiene el numero de instancias del clasificador
    outFrame = datos # Se crea el Frame resultante a partir del del Frame con los datos

    for i in range(num_pruebas): # Por cada instancia del clasificador
        p_colA = prueba.loc[i].values[0] # Se obtiene el valor de la Primera Columna
        p_colB = prueba.loc[i].values[1] # Se obtiene el valor de la Segunda Columna
        
        # Se realiza el calculo de la distancia y se agrega a una nueva columna
        outFrame[f'distance-of_test_{i+1}'] = (((p_colA - outFrame[colAName])**2) + ((p_colB - outFrame[colBName])**2))**0.5

    return outFrame # Se regresa el Frame resultante

# Funcion que calcula los vecinos y predice la clase asignada,
def NeighboursAndClass(result,k,prueba): # Recibe el Frame resultante de la funcion anterior, el valor de k establecido por el usuario y los datos de clasificacion
    num_pruebas = prueba.shape[0] # Se obtiene el numero de instancias del clasificado
    colClass = result.columns[2] # Nombre de la Columna con la Clase

    result_temp = result # Se crea un frame temporal para mostrar los datos del frame resultante

    print('A continuacion se mostraran los vecinos y la clase asignada a cada prueba')
    for i in range(num_pruebas): # Por cada instancia del clasificador
        print(f'Vecinos Prueba {i+1}')
        print('ID    Class   ')
        result_temp = result_temp.sort_values(f'distance-of_test_{i+1}') # Se ordenan de menor a mayor las distancias para calcular las mas cercanas
        print(result_temp.iloc[:k,2]) # Se muestra en pantalla los vecinos segun el k establecido
        print('Prediccion de Clase Final:',result_temp.iloc[:k][colClass].max()) # Se muestra la predicción de la clase asignada
        print()

def ExitFiles(result,prueba): # Funcion que crea archivo csv a partir del frame con las distnacias obtenidas
    colAName = result.columns[0] # Nombre de la Primera Columna
    colBName = result.columns[1] # Nombre de la Segunda Columna
    colClass = result.columns[2] # Nombre de la Columna con la Clase

    exitFrame = result.drop([colAName,colBName,colClass],axis=1) # Se genera un frame unicamente con las distancias

    exitFrame.to_csv(f'Salida.csv') # Se crea el archivo

# MAIN
testfile = 'Entrenamiento.csv' # Se llama al archivo con los datos de Entrenamiento
classfile = 'Clasificacion.csv' # Se llama al archivo con los datos de la Clasificacion

try: # Se aseugra que se lea el archivo correctamente
    datos = FrameFromCSV(testfile) # Se llama a la funcion para leer el archio y crear el frame
except:
    print('NO SE PUDO LEER EL ARCHIVO DE ENTRENAMIENTO')

try: # Se aseugra que se lea el archivo correctamente
    prueba = FrameFromCSV(classfile) # Se llama a la funcion para leer el archio y crear el frame
except:
    print('NO SE PUDO LEER EL ARCHIVO DE CLASIFICACION')

result = EuclidicKNN(datos,prueba) # Se llama a la funcion para generar el frame con las distancias
#print(result)

while True: # Ciclo infinito para asegurar un correcto input
    try:
        k = int(input('Ingrese el valor para el coeficiente euclidiano: ')) # Valor de k ingresado por el usuario
        break
    except:
        print('VALOR INVALIDO')

NeighboursAndClass(result,k,prueba) # Llamada a la funcion para mostrar los vecinos y la prediccion de la clase

ExitFiles(result,prueba) # Llamada a la funcion que genera el archivo de salida