import numpy as np

import math as mt
import spicy as sp
from numpy import asarray
from calculo_imagenes import *
# Temporizador
from timeit import default_timer as timer
start = timer()
# importa una imagen y la replica en formato png para poder jugar con ella
archivo = "horse.bmp"
replica = lectura(archivo)
U = asarray(replica)
width, height = replica.size
# cómo se carga una imagen de origen?
# La imagen se carga a desde la carpeta imagenes en la variable archivo presente en la linea 8.

# Longitud de las imagenes en la coordenada respectiva [mm]

# N es el número de pixeles en el eje x de la imagen de origen
# M es el número de píxeles en el eje y de la imagen origen
M = height
N = width

# pixeles en el eje x y y de la imagen de origen

x = np.arange(0, N, 1)
y = np.arange(0, M, 1)

# n y m es un reposicionamiento de coordenadas x y y respectivamente para poner un cero en el centro de la imgen
n, m = np.meshgrid(x - (N/2), y - (M/2))


# calculo de delta x y y

dx = 0.005
dy = 0.005


# distancia en la que se posiciona la imagen [m]
lamb = 0.000633  # micrometros
k = 2*mt.pi/lamb
z = -450  # cada 10e3 corresponde a 1mm
dx_ = (lamb*z)/(N*dx)
dy_ = (lamb*z)/(M*dy)

# se muestra la imagen


# Calculo de la fase 1

fase1 = np.exp2((1j * mt.pi / (lamb * z)) *
                (np.power(n * dx, 2) + np.power(m * dy, 2)))
fase1 = U*fase1

mostrar(np.abs(fase1), "fase1", "pixeles en el eje x", "pixeles en el eje y")
# Calculo de la transformada de Fourier

campo = np.fft.fftshift(fase1)
mostrar(np.abs(campo), "fft shift 1",
        "pixeles en el eje x", "pixeles en el eje y")
ft = sp.fft.fft2(campo)
mostrar(np.abs(ft), "fft", "pixeles en el eje x", "pixeles en el eje y")
ft = np.fft.fftshift(ft)


fase2 = np.exp2(1j*2*mt.pi*z/lamb)/(1j*lamb*z)
fase3 = np.exp2((1j * mt.pi / (lamb * z)) *
                (np.power(n * dx_, 2) + np.power(m * dy_, 2)))

final = dx*dy*fase2*fase3*ft


mostrar(np.log(np.abs(final)), "Imagen final",
        "pixeles en el eje x", "pixeles en el eje y")
print("without GPU:", timer()-start)
