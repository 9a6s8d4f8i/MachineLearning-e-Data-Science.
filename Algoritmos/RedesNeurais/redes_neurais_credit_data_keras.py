# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:42:35 2019

@author: felip_000
"""


import pandas as pd
base = pd.read_csv('credit-data.csv')
base.describe()

valor = base['age'][base.age>0].mean()
base.loc[base.age<0, 'age'] = 40.92770044906149

previsores = base.iloc[:, 1:4].values
    

classe = base.iloc[:,4].values
    

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis =0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores) #Valores escalonados

###########################CRIAÇÃO BASE DE TESTE###############################

from sklearn.model_selection import train_test_split
previsores_treinamento,previsores_teste, classe_treinamento,classe_teste  = train_test_split(previsores,classe,test_size=0.25,random_state=0)

import keras 
from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()

# camada oculta com 2 neuro utilizando relu como metodo de ativação, 3 entradas
classificador.add(Dense(units = 2, activation='relu', input_dim = 3))

# Outra camada oculta
classificador.add(Dense(units=2, activation='relu'))

# Camada de saída
classificador.add(Dense(units=1, activation='sigmoid'))
classificador.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size= 10, epochs=100)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)


from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)












