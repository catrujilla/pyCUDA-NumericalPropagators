import numpy as np
import math as mt
from numpy import asarray
from PIL import Image
import spicy as sp
from calculo_imagenes import *

#iniciar CUDA
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import cupy as cp
from reikna.fft import FFT,FFTShift

dev = drv.Device(0)

#importa una imagen y la replica en formato png para poder jugar con ella
archivo="punto.png"
replica=lectura(archivo)
U = asarray(replica)
width, height = replica.size

#Longitud de las imagenes en la coordenada respectiva [m]


#N es el número de píxeles en el eje x de la imagen origen

N=height

#M es el número de pixeles en el eje y de la imagen de origen

M=width

# Definición vectorial de los dx y dy
x= np.arange(0, N, 1)
y= np.arange(0, M, 1)

#P es el numero de píxeles en el eje x de A
#Q es el numero de píxeles en el eje y de A
P,Q=np.meshgrid(x -(N/2), y - (M/2))



#calculo de delta x y y
dx=0.0001
fx=1/(dx*N)
dy=0.0001
fy=1/(dy*M)

# distancia en la que se posiciona la imagen [m]

z=0.001

# Numéro y longitud de onda

lamb=0.000005
k=(2*mt.pi/lamb)



# ¿Cómo implementar fft en CUDA
reikna_array = gpuarray.to_gpu(U)

fft= FFTShift(reikna_array)

fft=FFT(fft)

fft= FFTShift(fft)



#operar con la fft de salida

exp=np.exp(1j*z*np.pi*np.sqrt(np.power(k,2)+np.power(P*fx,2)+np.power(Q*fy,2)))


reikna_array = gpuarray.to_gpu(exp)
casi= gpuarray.dot(fft,exp)

final= casi.get()

final = np.fft.fftshift(final)
final= np.fft.ifft2(final)
final = np.fft.fftshift(final) 

print(final)
fase_=fase(final)
amplitud_= amplitud(final)
intensidad_=intensidad(final)

#calculo de la matriz normalizada
imagen = Image.fromarray(intensidad_)

# guardar la imagen en formato PNG
imagen.save('imagenes/fase.png')