# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:33:30 2019

@author: felip_000
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

#Teste
# =============================================================================
# rede = buildNetwork(2, 3 , 1, outclass = SoftmaxLayer,
#                    bias = False,
#                    hiddenclass = SigmoidLayer)
# 
# print(rede['in'])
# print(rede['hidden8'])
# print(rede['out'])
# print(rede['bias'])
# =============================================================================

#(duas entradas, 3 ocultas, 1 saída)
rede = buildNetwork(2,3,1)
#(duas entradas, uma saída)
base = SupervisedDataSet(2,1)
base.addSample((0,0), (0,))
base.addSample((0,1), (1,))
base.addSample((1,0), (1,))
base.addSample((1,1), (0,))
#print(base['input'])
#print(base['target'])

#treinamento do algoritmo, com a rede, a base de dados, a taixa de aprendizagem e o momento
treinamento = BackpropTrainer(rede, dataset= base, learningrate= 0.01, momentum = 0.06)

for i in range(1,30000):
    erro = treinamento.train()
    if i % 1000 ==0:
        print("Erro: %s" % erro)

print(rede.activate([0,0]))
print(rede.activate([0,1]))
print(rede.activate([1,0]))
print(rede.activate([1,1]))   