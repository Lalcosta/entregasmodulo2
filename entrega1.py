'''
Eduardo Acosta Hernández A01375206
Algoritmo implementado: Regresión lineal
Dataset: Fish market
Recuperado de: https://www.kaggle.com/datasets/aungpyaeap/fish-market/code
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

    
#E(y/x)=B0+B1*x
def linear_regresion(x1,y1,predict):
    print("\n -------------------------------------------\n")
    print(f"Variable x={x1.name} \nVariable y={y1.name}")
    x=x1.values
    y=y1.values
    print(f'Coeficiente de correlación entre las variables: {np.corrcoef(x,y)}')
    #Total de filas: 159
    #31 datos para test (20% aprox)
    x_test=x[31:]
    y_test=y[31:]
    
    #128 datos para train (80% aprox)
    x=x[:128]
    y=y[:128]
    
    
    #Promedios
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    

    #Cálculo de B0 y B1
    num=0
    den=0
    for i in range(len(x)):
        num += (x[i] - x_mean) * (y[i] - y_mean)
        den += (x[i] - x_mean) ** 2
    b1 = num/den
    b0 = y_mean - (b1 * x_mean)
    
    #Imprimimos la función de regresión
    print(f'La función de regresión es y = {b0} + {b1}x')
    
    #Evaluamos el modelo mediante R2 con los datos de training
    t_train=0
    r_train=0
    for i in range(len(x)):
        y_pred_train = b0 + b1 *x[i]
        t_train += (y[i]-y_mean)**2
        r_train += (y[i]-y_pred_train)**2
    r2 = 1 -(r_train/t_train)
    print(f'El coeficiente R2 con los datos de entrenamiento es igual a {r2}')
    
    #Evaluamos el modelo mediante R2 con los datos de testing
    t_test=0
    r_test=0
    for i in range(len(x_test)):
        y_pred_test = b0 + b1 *x_test[i]
        t_test += (y_test[i]-y_mean)**2
        r_test += (y_test[i]-y_pred_test)**2
    r2 = 1 -(r_test/t_test)
    print(f'El coeficiente R2 con los datos de prueba es igual a {r2}')
    
    #Realizamos las predicciones de los valores solicitados
    for i in predict:
        print(f'La predicción del valor {i} es: {b0+b1*i}')
        
    #Generamos gráfica de los datos con la linea de regresión
    
    #Generamos valores para la gráfica
    x_graf=np.linspace(np.min(x)-x_mean,np.max(x)+x_mean,len(x))
    
    #Función de regresión
    y_graf = b0 + b1 * x_graf
    
    #Linea de regresión
    plt.plot(x_graf,y_graf,color="red",label="Linea de regresión")
    
    #Gráfica de dispersión de los datos
    plt.scatter(x,y,color="blue",label="Dispersión de los datos")
    plt.xlabel(x1.name)
    plt.ylabel(y1.name)
    plt.legend()
    plt.show()
    print("\n -------------------------------------------\n")
    
    

    
    
df = pd.read_csv("fish.csv")
dataplot = sb.heatmap(df.corr(), cmap="YlGnBu", annot=True) 
plt.show() 
#print(df.info())

"""
Probamos con variables Weight y Lenght3 que tienen una correlación fuerte en el dataset
"""
linear_regresion(df['Weight'],df['Length3'],[100,400,500,648,1300])

"""
Probamos con variables Height y Lenght1 que tienen una correlación baja en el dataset
"""
linear_regresion(df['Height'],df['Length1'],[4,12,20,27])

"""
Probamos con variables Height y Weight que tienen una correlación media en el dataset
"""
linear_regresion(df['Height'],df['Weight'],[10,15,0,22,30])