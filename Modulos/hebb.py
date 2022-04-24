import math
import numpy as np

#función signo
def signo(z):
    if z == 0:
        return 0
    elif z > 0:
        return 1
    return -1
# versión vectorizada de la función signo
sig = np.vectorize(signo)

#clase red neuronal
class Perceptron():
    '''
    RedNeuronal(capas = [28*28, 10])
    .aprender(datos, razonApr = 1)
    .computar(i)
    '''
    def __init__(self, capas = [28*28, 10]):
        self.pesos = np.random.randn(capas[1], capas[0])
        self.umbrales = np.random.randn(capas[1], 1)
        self.capas = capas
    
    def aprender(self, datos, razonApr = 1):
        for d in datos:
            x, o = d
            y = sig(np.dot(self.pesos, x) + self.umbrales)
            if (y != o).any():
                self.pesos = self.pesos + razonApr * np.dot(x, o.T).T
    
    def computar(self, x):
        return sig(np.dot(self.pesos, x) + self.umbrales)
