# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:11:24 2019

@author: felip_000
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = FeedForwardNetwork()

camadaEntrada = LinearLayer(2)
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)
bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

conexaoEntradaOculta = FullConnection(camadaEntrada, camadaOculta)
conexaoOcultaSaida = FullConnection(camadaOculta, camadaSaida)
conexaoBias1 = FullConnection(bias1, camadaOculta)
conexaoBias2 = FullConnection(bias2, camadaSaida)

rede.sortModules()

print(rede)
print(conexaoEntradaOculta.params)
print(conexaoOcultaSaida.params)
print(conexaoBias1.params)
print(conexaoBias2.params)