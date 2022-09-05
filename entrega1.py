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
def linear_regresion(x1,y1):
    x=x1.values
    y=y1.values
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
    
    #Evaluamos el modelo mediante R2
    t=0
    r=0
    for i in range(len(x)):
        y_pred = b0 + b1 *x[i]
        t += (y[i]-y_mean)**2
        r += (y[i]-y_pred)**2
    r2 = 1 -(r/t)
    print(f'El coeficiente R2 es igual a {r2}')
    #Generamos gráfica de los datos con la linea de regresión
    
    #Generamos valores para la gráfica
    x_graf=np.linspace(np.min(x)-100,np.max(x)+300,len(x))
    
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
    

df = pd.read_csv("fish.csv")
dataplot = sb.heatmap(df.corr(), cmap="YlGnBu", annot=True) 
plt.show() 
print(df.info())
"""
Identificamos que las variables de Weight y Lenght tienen una correlación positiva por lo que a partir de
"""
linear_regresion(df['Weight'],df['Length3'])