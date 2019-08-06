# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 18:32:02 2019

@author: felip_000
"""

import pandas as pd
base = pd.read_csv('risco-credito.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

from sklearn.tree import DecisionTreeClassifier, export

classificador = DecisionTreeClassifier(criterion="entropy")

classificador.fit(previsores,classe)

export.export_graphviz(classificador, 
                       feature_names =["historia", "divida", "garantias", "renda"],
                       class_names = ["alto", "moderado", "baixo"],
                       filled = True,
                       leaves_parallel = True
                       )

resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])

print(classificador.classes_)#Quais são as classes
print(classificador.class_count_)#Quantos itens de cada classe
print(classificador.class_prior_)#probabilidade a priori de cada classe




