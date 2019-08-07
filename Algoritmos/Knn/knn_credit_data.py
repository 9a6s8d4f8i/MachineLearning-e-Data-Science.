# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 18:06:49 2019

@author: felip_000
"""

import pandas as pd
base = pd.read_csv("credit-data.csv")

menores = base.loc[base.age<0]
base.loc[base.age<0,"age"] = 40.92


previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

#No algoritmo KNN é necessário fazer um escalonamento.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste,  classe_treinamento , classe_teste  = train_test_split(previsores,classe,test_size=0.25,random_state=0)


from sklearn.neighbors import KNeighborsClassifier

classificador = KNeighborsClassifier(n_neighbors=5,metric="minkowski", p = 2)
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)


import collections
collections.Counter(classe_teste)






