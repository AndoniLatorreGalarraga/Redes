import numpy as np
import random
from Modulos import redNeuronal as rn
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
    
    datosImagenes, datosEtiquetas, testImagenes, etiquetas= mnist.obtenerDatos()
    datosEntr = [(i, o) for i, o in zip(datosImagenes, datosEtiquetas)]

    print('Perceptron:')
    perceptron = per.Perceptron()
    test(perceptron, testImagenes, etiquetas)
    entrenar(perceptron, 20, 60000, 1.5)

    print('Red:')
    red = rn.RedNeuronal()
    test(red, testImagenes, etiquetas)
    entrenar(red, 20, 10, 1.5)
 
def result(array):
    list = array.T.tolist()[0]
    return list.index(max(list))

if __name__ == '__main__':
    main()