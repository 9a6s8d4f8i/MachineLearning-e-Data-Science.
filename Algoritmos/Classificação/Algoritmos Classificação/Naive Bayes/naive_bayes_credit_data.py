# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
base = pd.read_csv('credit-data.csv')


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


from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste,  classe_treinamento , classe_teste  = train_test_split(previsores,classe,test_size=0.25,random_state=0)

from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()

classificador.fit(previsores_treinamento, classe_treinamento)
resposta = classificador.predict(previsores_teste)
    
from sklearn.metrics import confusion_matrix, accuracy_score

#Devolve a probabilidade de acertos do algoritmo de classificação 
precisão = accuracy_score(classe_teste,resposta)
#Devolve a matriz de confusão
matriz = confusion_matrix(classe_teste,resposta)












