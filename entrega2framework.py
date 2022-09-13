'''
Eduardo Acosta Hernández A01375206
Algoritmo implementado: Decision Tree Classifier
Dataset: Fish market
Recuperado de: https://www.kaggle.com/datasets/aungpyaeap/fish-market/code
'''

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier


dataFrame = pd.read_csv("Fish.csv")

print(dataFrame.head())

X=dataFrame.drop('Species',axis=1)
y=dataFrame['Species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)
dt = DecisionTreeClassifier(random_state=23)
dt.fit(X_train,y_train)
print("Exactitud del modelo", dt.score(X_test,y_test))
print("Exactitud del modelo(train)",dt.score(X_train,y_train))

#Pruebas con el set de test
print("Prueba 1\nParametros:")
print(X_test.iloc[0])
print("Resultado:")
print(dt.predict([X_test.iloc[0]]))

print("Prueba 2\nParametros:")
print(X_test.iloc[0])
print("REsultado:")
print(dt.predict([X_test.iloc[0]]))

print("Prueba 2\nParametros:")
print(X_test.iloc[0])
print("Resultado:")
print(dt.predict([X_test.iloc[0]]))

print("Prueba 1\nParametros:")
print(X_test.iloc[0])
print("Resultado:")
print(dt.predict([X_test.iloc[0]]))

print("Predicción 1")
print(dt.predict([[67,19.6,19,22.8,6.78,5.62]]))

print("Predicción 2")
print(dt.predict([[191,30.8,29,17,8.3,7.41]]))

print("Predicción 3")
print(dt.predict([[248,27.7,31,24,12,3]]))