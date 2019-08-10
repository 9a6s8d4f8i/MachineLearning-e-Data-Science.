# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:24:22 2019

@author: felip_000
"""

import pandas as pd

base = pd.read_csv("census.csv")

#Base de dados não possui valores inconsistentes

previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values


#Como em machine learn os algoritmos geralmente utilizam dados numéricos para 
#devolver um resultado, é necessário transformar os atributos categóricos em 
#atributos discretos, utiliza-se LabelEncoder pra isso

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

labelEncoder_previsores = LabelEncoder()

#Atributo da coluna 1 é categórico
previsores[:,1] = labelEncoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelEncoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelEncoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelEncoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelEncoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelEncoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelEncoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelEncoder_previsores.fit_transform(previsores[:,13])

#Existe uma ineficiência nessa solução, pois essas variáveis trasnformadas são do tipo nominal
#No caso não posso dizer por exemplo que uma raça é melhor que outra

onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

labelEncoder_classe = LabelEncoder()

classe = labelEncoder_classe.fit_transform(classe)


standardScaler = StandardScaler()
previsores = standardScaler.fit_transform(previsores)

###########################CRIAÇÃO BASE DE TESTE###############################

from sklearn.model_selection import train_test_split
previsores_treinamento,previsores_teste,classe_treinamento, classe_teste  = train_test_split(previsores,classe,test_size=0.15,random_state=0)










