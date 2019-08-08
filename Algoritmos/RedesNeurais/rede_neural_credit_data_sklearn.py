# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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



from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste,classe_treinamento,classe_teste  = train_test_split(previsores,classe,test_size=0.25,random_state=0)





from sklearn.neural_network import MLPClassifier

classificador = MLPClassifier(max_iter=1000,
                              verbose = True,
                              tol=0.0000010,
                              solver = 'adam',
                              hidden_layer_size=(100),
                              activation='relu')


classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)


from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)









