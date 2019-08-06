# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:24:22 2019

@author: felip_000
"""

import pandas as pd

base = pd.read_csv("census.csv")

previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values


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


onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

labelEncoder_classe = LabelEncoder()

classe = labelEncoder_classe.fit_transform(classe)


standardScaler = StandardScaler()
previsores = standardScaler.fit_transform(previsores)


from sklearn.cross_validation import train_test_split
previsores_treinamento,previsores_teste,classe_treinamento, classe_teste  = train_test_split(previsores,classe,test_size=0.15,random_state=0)


from sklearn.tree import DecisionTreeClassifier

classificador = DecisionTreeClassifier(criterion="entropy", random_state=0)

classificador.fit(previsores_treinamento, classe_treinamento)
resposta = classificador.predict(previsores_teste)


from sklearn.metrics import accuracy_score, confusion_matrix

precisao = accuracy_score(classe_teste, resposta)
matriz = confusion_matrix(classe_teste, resposta)




