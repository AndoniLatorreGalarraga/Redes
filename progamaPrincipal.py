import numpy as np
import random, time
from Modulos import redNeuronalSig as rnsigma
from Modulos import redNeuronalReLU as rnrelu
from Modulos import hebb as per
from Modulos import mnist


def main():
    def entrenar(red, rondas, tamaño, razon):
        lonDatosEntr  = len(datosEntr)
        for n in range(rondas):
            random.shuffle(datosEntr)
            for i in range(0, lonDatosEntr, tamaño):
                red.aprender(datosEntr[i:i+tamaño], razonApr = razon)
                print('%', round(100*(i+1)/lonDatosEntr, 2), '      ', end='\r')
            print('Progreso: ', n + 1, '/', rondas, sep = '')
            test(red, testImagenes, etiquetas)

    def test(red, testImagenes, etiquetas):
        correctas, total = 0, 0
        for (imagen, etiqueta) in zip(testImagenes, etiquetas):
            if result(red.computar(imagen)) == etiqueta:
                correctas += 1
            total += 1
        print('Corectas: ', correctas, '/', total, sep = '')
    

    print('Perceptron:')
    datosImagenes, datosEtiquetas, testImagenes, etiquetas= mnist.obtenerDatosPer()
    datosEntr = [(i, o) for i, o in zip(datosImagenes, datosEtiquetas)]
    perceptron = per.Perceptron()
    test(perceptron, testImagenes, etiquetas)
    entrenar(perceptron, 2, 60000, 0.01)

    print('Red:')

    datosImagenes, datosEtiquetas, testImagenes, etiquetas= mnist.obtenerDatosRed()
    datosEntr = [(i, o) for i, o in zip(datosImagenes, datosEtiquetas)]

    red = rnsigma.RedNeuronal()
    test(red, testImagenes, etiquetas)
    t0 = time.time()
    entrenar(red, 2, 10, 1.5)
    print('tiempo', time.time()- t0)

def result(array):
    list = array.T.tolist()[0]
    return list.index(max(list))

if __name__ == '__main__':
    main()