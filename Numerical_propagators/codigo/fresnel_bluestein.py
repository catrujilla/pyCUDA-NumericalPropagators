import numpy as np
import cmath as cmt
import math as mt
import spicy as sp
from numpy import asarray
from PIL import Image
from calculo_imagenes import *
# importa una imagen y la replica en formato png para poder jugar con ella
from timeit import default_timer as timer
archivo = "die_1.jpg"
replica = lectura(archivo)
U = asarray(replica)
print(U.shape, " forma de U")
width, height = replica.size
# cómo se carga una imagen de origen?
# Longitud de las imagenes en la coordenada respectiva [m]


# N es el número de píxeles en el eje x de la imagen origen

M = height

# M es el número de pixeles en el eje y de la imagen de origen

N = width

# pixeles en el eje x y y de la imagen de origen

M_ = height
N_ = width
x = np.arange(0, N, 1)
y = np.arange(0, M, 1)

# n,m son coordenadas de imagen inicial.

n, m = np.meshgrid(x - (N/2), y - (M/2))

# Q es el numero de píxeles en el eje y de la imagen final


# calculo de delta x y y
dx = 7e-6
dy = 7e-6


# distancia en la que se posiciona la imagen [m]
lamb = 632e-9
k = 2*mt.pi/lamb
z = 30/100

# diferencial en el caso de la imagen final

dx_ = 4.5e-5
dy_ = 4.5e-5

# Definición de matrices de relleno.

padx = int(N/2)
pady = int(M/2)

# Enseña la imagen inicial

mostrar(U, "Imagen de entrada",
        "pixeles en el eje x", "pixeles en el eje y")
start = timer()
# Calculo del termino de la fase 1 que será multiplicado con la imgan de entrada.

fase1 = dx*dy*np.exp2(1j*mt.pi/(lamb*z)*((dx*(dx-dx_) *
                      (np.power(n, 2)))+(dy*(dy-dy_)*(np.power(m, 2)))))
fase1 = np.pad(fase1*U, ((padx, padx), (pady, pady)), mode='constant')

# Se enseña el resultado


# Se aplica la transformada rápida de fourier

campo = np.fft.fftshift(fase1)
ft = np.fft.fft2(campo)
ft = np.fft.fftshift(ft)

mostrar((np.abs(ft)), "final",
        "pixeles en el eje x", "pixeles en el eje y")
# Se calcula la segunda fase

fase2 = np.exp2(1j*mt.pi/(lamb*z) *
                ((dx*dx_*(np.power(n, 2)))+(dy*dy_*(np.power(m, 2)))))
print(fase2.shape)
fase2 = np.pad(fase2, ((padx, padx), (pady, pady)), mode='constant')
print(fase2.shape)
mostrar((np.abs(fase2)), "si la vida te da limones",
        "pixeles en el eje x", "pixeles en el eje y")
# Transformada de Fourier rápida aplicada en la segunda fase

campo2 = np.fft.fftshift(fase2)
mostrar((np.abs(campo2)), "si la vida te da limones",
        "pixeles en el eje x", "pixeles en el eje y")
ft2 = np.fft.fft2(campo2)
ft2 = np.fft.fftshift(ft2)

mostrar((np.abs(ft2)), "has limonada ",
        "pixeles en el eje x", "pixeles en el eje y")
# Se calcula la multiplicación entre transformadas de fourier

inv = ft*ft2

# Transformadas inversas rápidas de fourier

inv = np.fft.fftshift(inv)
inv = np.fft.ifft2(inv)
inv = np.fft.fftshift(inv)


# Calculo y ajustes finales

inv = inv[padx:padx + N, pady:pady + M]

# Calculo de las últimas dos fases

fase3 = np.exp2(-1j*mt.pi*((dx_*(dx-dx_)*(np.power(n, 2))) +
                (dy_*(dy-dy_)*(np.power(m, 2)))/(lamb*z)))
fase4 = (np.exp2(1j*k*z))/(1j*lamb*z)

# Operaciones finales

final = inv*fase4
final = fase3*final
print(final.shape, "forma final")
print("without GPU:", timer()-start)
mostrar(np.log(np.abs(final)), "imagen de salida",
        "pixeles en el eje x", "pixeles en el eje y")
