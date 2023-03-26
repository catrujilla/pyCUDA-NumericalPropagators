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

mostrar(U)
