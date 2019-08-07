# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
base = pd.read_csv('credit-data.csv')
base.describe()

#################################INCONSISTÊNCIA#################################

#Técnicas pra tratar valores inconsistentes

base.loc[base.age<0] #Clientes com idades menores que 0.

#TECNICA 1
#Excluindo a tabela age. Não é uma solução boa
  #Comando exclui a coluna age inteira (1) sem retorno da coluna (implace)
base.drop('age',1,inplace = True)

#TECNICA 2
#Excluindo apenas os dados inconsistentes
   #Exclui as linhas com dados inconsistentes
base.drop(base[base.age<0].index,inplace = True)

#TECNICA 3
#Pedindo para o cliente informar os dados corretos.

#TECNICA 4
#Substituindo o valor dos dados inconsistentes pela média dos dados sem esses valores
valor = base['age'][base.age>0].mean()
base.loc[base.age<0, 'age'] = 40.92770044906149



#################################VALORES FALTANTES#############################

##Tecnicas pra tratar valores faltantes:

#Separo primeiro os dados:
     
    #Uma tabela para os previsores, (:)indica todas as linhas e (1:4) indica 
    #da coluna 1 até a 3
previsores = base.iloc[:, 1:4].values
    
    #Outra para a classe, resultado (4) indica que é só a coluna 4

classe = base.iloc[:,4].values
    

from sklearn.preprocessing import Imputer
#Completar valores faltantes
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis =0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])


#################################ESCALONAMENTO#################################
#Nesse caso, os valores de uma coluna são muito distantes dos valores de outra
#Por isso é necessário fazer um escalonamento dos atributos
#Um dos tipos de escalonamento é o StandartScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores) #Valores escalonados

###########################CRIAÇÃO BASE DE TESTE###############################

from sklearn.model_selection import train_test_split
classe_teste,previsores_teste,classe_treinamento,previsores_treinamento  = train_test_split(previsores,classe,test_size=0.25,random_state=0)


#












