import numpy as np
import math as mt
from calculo_imagenes import *
from timeit import default_timer as timer
#importa una imagen y la replica en formato png para poder jugar con ella
archivo="+3cm.tif"
replica=lectura(archivo)
U = np.array(replica)
mostrar(U)
start = timer()
width, height = replica.size
#M es el número de pixeles en el eje y de la imagen de origen
M=height

#N es el número de píxeles en el eje x de la imagen origen

N=width

#Definición de un array para antidad de puntos en cada eje

x= np.arange(0, N, 1)
y= np.arange(0, M, 1)

#P es el numero de píxeles en el eje x de la imagen final
#Q es el numero de píxeles en el eje y de la imagen final
P,Q=np.meshgrid(x -(N/2), y - (M/2))

#calculo de fx y fy
dx=0.0069
dy=0.0069
fx=1/(dx*N)
fy=1/(dy*M)
# distancia en la que se posiciona la imagen [mm]
lamb=0.000633 #m
z= -100   # en m

# Calculo de A(n,m,0) y A(n,m,z)

campo = np.fft.fftshift(U)
print("without GPU:", timer()-start)	
ft=np.fft.fft2(campo)
ft = np.fft.fftshift(ft)

#mostrar(np.log(np.abs(ft)))
exp=np.exp2(1j*z*mt.pi*np.sqrt(np.power(1/lamb,2) - (np.power(fx*P,2) + np.power(Q*fy,2))))
#mostrar(np.log(np.abs(exp)))
final=ft*exp
#mostrar(np.log(np.abs(final)))

final = np.fft.fftshift(final)
final = np.fft.ifft2(final)
final = np.fft.fftshift(final)
print("without GPU:", timer()-start)	
mostrar((np.abs(final)*np.abs(final)))

