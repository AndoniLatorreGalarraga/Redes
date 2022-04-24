import numpy as np
import idx2numpy

def obtenerDatosRed():
    datosEtiquetas = idx2numpy.convert_from_file('Datos/train-labels.idx1-ubyte')
    datosImagenes = idx2numpy.convert_from_file('Datos/train-images.idx3-ubyte')
    testEtiquetas = idx2numpy.convert_from_file('Datos/t10k-labels.idx1-ubyte')
    testImagenes = idx2numpy.convert_from_file('Datos/t10k-images.idx3-ubyte')
    datosImagenesReshape = []
    testImagenesReshape = []
    for image in datosImagenes:
        datosImagenesReshape.append((1/255)*np.reshape(image,(784,1)))
    for image in testImagenes:
        testImagenesReshape.append((1/255)*np.reshape(image,(784,1)))
    datosEtiquetasArray = []
    for label in datosEtiquetas:
        array = [0]*10
        array[label] = 1
        datosEtiquetasArray.append(np.array([array]).T)
    return datosImagenesReshape, datosEtiquetasArray, testImagenesReshape, testEtiquetas

def obtenerDatosPer():
    datosEtiquetas = idx2numpy.convert_from_file('Datos/train-labels.idx1-ubyte')
    datosImagenes = idx2numpy.convert_from_file('Datos/train-images.idx3-ubyte')
    testEtiquetas = idx2numpy.convert_from_file('Datos/t10k-labels.idx1-ubyte')
    testImagenes = idx2numpy.convert_from_file('Datos/t10k-images.idx3-ubyte')
    datosImagenesReshape = []
    testImagenesReshape = []
    for image in datosImagenes:
        datosImagenesReshape.append((1/255)*np.reshape(image,(784,1)))
    for image in testImagenes:
        testImagenesReshape.append((1/255)*np.reshape(image,(784,1)))
    datosEtiquetasArray = []
    for label in datosEtiquetas:
        array = [-1]*10
        array[label] = 1
        datosEtiquetasArray.append(np.array([array]).T)
    return datosImagenesReshape, datosEtiquetasArray, testImagenesReshape, testEtiquetas