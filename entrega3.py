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
from mlxtend.evaluate import bias_variance_decomp
import numpy as np
from sklearn.model_selection import validation_curve

dataFrame = pd.read_csv("Fish.csv")

print(dataFrame.head())

X=dataFrame.drop('Species',axis=1)
y=dataFrame['Species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)
dt = DecisionTreeClassifier(random_state=23, max_depth=10)
dt.fit(X_train,y_train)
print("Exactitud del modelo", dt.score(X_test,y_test))
print("Exactitud del modelo(train)",dt.score(X_train,y_train))
train_prec =  []
eval_prec = []
max_deep_list = list(range(3, 23))

for deep in max_deep_list:
    arbol3 = DecisionTreeClassifier(criterion='entropy', max_depth=deep)
    arbol3.fit(X_train, y_train)
    train_prec.append(arbol3.score(X_train, y_train))
    eval_prec.append(arbol3.score(X_test, y_test))

# graficar los resultados.
plt.plot(max_deep_list, train_prec, color='r', label='entrenamiento')
plt.plot(max_deep_list, eval_prec, color='b', label='evaluacion')
plt.title('Grafico de ajuste arbol de decision')
plt.legend()
plt.ylabel('precision')
plt.xlabel('cant de nodos')
plt.show()


"""avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        dt, X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy(), 
        loss='0-1_loss',
        random_seed=123)

print('MSE: %.3f' % avg_expected_loss)
print('Bias: %.3f' % avg_bias)
print('Variance: %.3f' % avg_var)
"""