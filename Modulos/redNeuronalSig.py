import math
import numpy as np

#función sigmoide
def sigmoide(z):
    try: #cálculo normal de la función sigmoide
        return 1 / (1 + math.exp(-z))
    except OverflowError: #protección para errores de overflow
        if z > 0:
            return 1
        return 0

# versión vectorizada de la función sigmoide
sig = np.vectorize(sigmoide)

#derivada de la función sigmoide
def dsigma(z):
    return sigmoide(z) * (1 - sigmoide(z))

#versión vectorizada de la derivada
dsig = np.vectorize(dsigma)

#clase red neuronal
class RedNeuronal():
    '''
    RedNeuronal(capas = [28*28, 16, 10])
    .aprender(datos, razonApr = 1)
    .computar(i)
    '''
    def __init__(self, capas = [28*28, 16, 10]):
        self.pesos = [np.random.randn(n, m) for n, m in zip(capas[1:], capas[:-1])]
        self.umbrales = [np.random.randn(n, 1) for n in capas[1:]]
        self.capas = capas
        self.profundidad = len(capas)
    
    def aprender(self, datos, razonApr = 1):
        gradW = [np.zeros((n, m)) for n, m in zip(self.capas[1:], self.capas[:-1])]
        gradB = [np.zeros((n, 1)) for n in self.capas[1:]]
        lonDatos = 0
        for d in datos:
            i , o = d
            lonDatos += 1
            #propagación hacia delante
            a = i
            aLista, zLista = [a], [None]
            for pesos, umbrales in zip(self.pesos, self.umbrales):
                z = np.dot(pesos, a) + umbrales
                zLista.append(z)
                a = sig(z)
                aLista.append(a)
            #propagación hacia atras
            dW= [np.zeros((n, m)) for n, m in zip(self.capas[1:], self.capas[:-1])]
            dB = [np.zeros((n, 1)) for n in self.capas[1:]]
            dA = [np.zeros((n, 1)) for n in self.capas]
            dA[-1] = aLista[-1] - o
            dB[-1] = np.multiply(dA[-1], dsig(zLista[-1]))
            dW[-1] = np.dot(dB[-1], aLista[-2].T)
            for l in range(self.profundidad - 2, 0, -1):
                dA[l] = np.zeros((self.capas[l],1))
                for k in range(self.capas[l+1]):
                    scalar = np.asscalar((dA[l+1][k] * dsigma(zLista[l+1][k])))
                    dA[l] = dA[l] + np.array([self.pesos[l][k]]).T * scalar
                dB[l-1] = np.multiply(dA[l], dsig(zLista[l]))
                dW[l-1] = np.dot(dB[l-1], aLista[l-1].T)
            for l in range(0, self.profundidad-1):
                gradB[l] = gradB[l] + dB[l]
                gradW[l] = gradW[l] + dW[l]
        #actualizar pesos y umbrales
        for l in range(0, self.profundidad-1):
            self.pesos[l] = self.pesos[l] - (gradW[l] * (2*razonApr/lonDatos))
            self.umbrales[l] = self.umbrales[l] - (gradB[l] * (2*razonApr/lonDatos))
    
    def computar(self, i):
        for w, b in zip(self.pesos, self.umbrales):
            i = sig(np.dot(w, i) + b)
        return i
