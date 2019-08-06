# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 18:32:02 2019

@author: felip_000
"""

import pandas as pd
base = pd.read_csv('risco-credito.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values


from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])


#Faz o treinamento do algoritmo, geranto a tabela de probabilidades
# é necessário entretanto que seja inicialmente 
#feito um "encoder" pois o naive bayes não aceitar atributos categóricos.
classificador.fit(previsores,classe)

#Um cliente com os seguintes históricos possuem qual risco de crédito?
#história boa, divida alta, garantias nenuhuma, renda>35
#história ruim, divida alto, garantas adequadas, renda <15

classificador.predict([[0,0,1,2], [3,0,0,0]])

print(classificador.classes_)#Quais são as classes
print(classificador.class_count_)#Quantos itens de cada classe
print(classificador.class_prior_)#probabilidade a priori de cada classe




