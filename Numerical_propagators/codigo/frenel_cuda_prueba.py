import numpy as np
import numpy as np
import math as mt
from numpy import asarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from calculo_imagenes import *
from timeit import default_timer as timer

#importa una imagen y la replica en el formato de la imagen para su procesamiento
archivo="horse.bmp"
replica=lectura(archivo)
U = asarray(replica)
width, height = replica.size

#Se enseña la imagen de origen
mostrar(U,"original")

#Esta linea es para iniciar un timer que cuente cuanto demora en compilar el programa
start = timer()

#M es el número de píxeles en el eje y de la imagen origen
M=height

#N es el número de pixeles en el eje x de la imagen de origen
N=width

#pixeles en el eje x y y de la imagen de origen
x= np.arange(0, N, 1)
y= np.arange(0, M, 1)

#Dimensionamiento de la imagen numéricamente
n,m=np.meshgrid(x -(N/2), y - (M/2))

#Convertir los datos a formatos adecuados para el envío a la gpu
n=n.astype(np.float32)
m=m.astype(np.float32)

#calculo de delta x y y
dx=0.005
dy=0.005

# distancia en la que se posiciona la imagen [m]
lamb=0.000633 #micrometros
z= 450 # cada 10e3 corresponde a 1mm
dx_=(lamb*z)/(N*dx)
dy_=(lamb*z)/(M*dy)
pi=3.1415
#definición dimensiones de las variables en un entorno CUDA

U_gpu = cuda.mem_alloc(U.nbytes)

# definición valores de las variables en un entorno CUDA
cuda.memcpy_htod(U_gpu, U)
m_gpu=gpuarray.to_gpu(m)
n_gpu=gpuarray.to_gpu(n)


#Código de las funciones para la función
mod = SourceModule("""
#include <math.h>
#include <stdio.h>
#include <cuComplex.h>

__global__ void Fresnel(float *U_gpu,cuComplex *dest_copia, cuComplex *dest_gpu, float *m_gpu, float *n_gpu, float dx, float dy, float lamb, float z, float pi, int N, int M)
{
    int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
    int i2= ((fila*N)+col);

	dest_gpu[i2].x = U_gpu[i2];
    dest_gpu[i2].y = U_gpu[i2];
}	
__global__ void fft_shift(cuComplex *final,cuComplex *dest_gpu, int N, int M)
{
	int n2 = N / 2;
	int m2 = M / 2;

	//Descriptores de cada hilo
    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
	int fila2 = fila + m2;
	int col2 = col + n2;
    
    final[(fila2*N + col2)].x = dest_gpu[(fila*N+col)].x;  //Guardo el primer cuadrante
	final[(fila*N+col)].x = dest_gpu[(fila2*N+col2)].x;  //en el primer cuadrante estoy poniendo lo que hay en el tercero
    final[(fila2*N + col2)].y = dest_gpu[(fila*N+col)].y;  //Lo mismo pero para los imaginarios
	final[(fila*N+col)].y = dest_gpu[(fila2*N+col2)].y; 
    
    final[(fila*N + col2)].x = dest_gpu[(fila2*N+col)].x;  //Guardo el segundo cuadrante
	final[(fila2*N+col)].x = dest_gpu[(fila*N+col2)].x;  //en el segundo cuadrante estoy poniendo lo que hay en el tercer cuadrante
    final[(fila*N + col2)].y = dest_gpu[(fila2*N+col)].y;  //Lo mismo pero para los imaginarios
	final[(fila2*N+col)].y = dest_gpu[(fila*N+col2)].y;  
}

__global__ void fft_shift_img(float *final,float *U_gpu, int N, int M)
{
	int n2 = N / 2;
	int m2 = M / 2;

	//Descriptores de cada hilo
    int fila = blockIdx.x*blockDim.x+ threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	int fila2 = blockIdx.x*blockDim.x + threadIdx.x + m2;
	int col2 = blockIdx.y*blockDim.y + threadIdx.y + n2;
    final[fila*N+col] = U_gpu[((fila2*N) + (col2))/4];
    final[fila2*N+col2] = U_gpu[(fila*N + col)/4];
    final[fila*N+col2] = U_gpu[(fila2*N + col)/4];
    final[fila2*N+col] = U_gpu[(fila*N + col2)/4];
}
    
__global__ void Fresnel2(cuComplex *final,cuComplex *dest_copia, cuComplex *dest_gpu, float *m_gpu, float *n_gpu,  float dx_, float dy_, float lamb, float z, float pi, int N, int M,float dx, float dy)
{

    int fila = blockIdx.x*blockDim.x+threadIdx.x;
	int col = blockIdx.y*blockDim.y+threadIdx.y;
    int i2= ((fila*N)+col);

    dest_copia[i2].x = __cosf((2*pi*z/lamb)+(pi / (lamb * z) * ((pow(n_gpu[i2] * dx_, 2) + (pow(m_gpu[i2] * dy_, 2))))));
    dest_copia[i2].y = __sinf((2*pi*z/lamb)+(pi / (lamb * z) * ((pow(n_gpu[i2] * dx_, 2) + (pow(m_gpu[i2] * dy_, 2))))));
    dest_gpu[i2].x = (-1)*((final[i2].y * dest_copia[i2].x) + (final[i2].x * dest_copia[i2].y))/(lamb*z);
    dest_gpu[i2].y = (-1)*((final[i2].x * dest_copia[i2].x) - (final[i2].y * dest_copia[i2].y))/(lamb*z);

}
""")
#Se llaman las funciones de la GPU para guardarlas en la CPU 
Fresnel=mod.get_function("Fresnel")
fft_shift=mod.get_function("fft_shift")
Fresnel2=mod.get_function("Fresnel2")

#Esta función solamente shiftea imagenes no complejas
#fft_shift_image=mod.get_function("fft_shift_img")

#Variables de uso a lo largo de las funciones
dest_copia=gpuarray.empty((N,M), np.complex64)
dest_gpu= gpuarray.empty((N,M), np.complex64)
x_gpu = gpuarray.empty((N,M), np.float32)
final = gpuarray.empty((N,M), np.complex64)

#Con el grid_dim se puede ajustar la malla que se va a considerar
block_dim=(16,16,1)

#Mallado para la primera correción de fase
grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)

#Se aplica la primera función
Fresnel((U_gpu),dest_gpu,dest_copia, (m_gpu), (n_gpu), np.float32(dx), np.float32(dy),np.float32(lamb),np.float32(z),np.float32(pi), np.int32(N),np.int32(M), block=(16,16,1), grid=grid_dim)

#Se llama mai como una variable de transporte de datos de la GPU a la CPU
mai=dest_gpu.get()

#Se dimensiona la variable mai
finale=mai.reshape((M,N))

#Imagen de salida
mostrar(((np.abs(finale))),"Primer corregimiento de fase")

#En esta parte se va a aplicar otra función por lo que se debe definir un grid adecuado para lograrlo
grid_dim = (M // (2*block_dim[0]), N// (2*block_dim[1]),1)

#Función de shifteo para pasar a la 
fft_shift(final,dest_gpu,np.int32(N),np.int32(M),block=(16,16,1), grid=grid_dim)

#La variable mai es un nombre temporal de recuperación
mai=final.get()

#Se recrea la matriz de salida de el shift en la cpu
finale=mai.reshape((M,N))

#Imagen de salida
mostrar((np.log10(np.abs(finale))),"fft shift")


#Transformada de Fourier
plan = cu_fft.Plan(final.shape, np.complex64, np.complex64)
cu_fft.fft(final, dest_gpu, plan)

#La variable mai es un nombre temporal de recuperación
mai=dest_gpu.get()

#Se recrea la matriz en la cpu
finale=mai.reshape((M,N))

#Imagen de salida para Fourier
mostrar((np.log10(np.abs(finale))),"FFT")

#Se inicia la funcióm de shifteo una vez se hace la transformada de Fourier
fft_shift(final,dest_gpu,np.int32(N),np.int32(M),block=(16,16,1), grid=grid_dim)

#La variable mai es un nombre temporal de recuperación
mai=final.get()

#Se recrea la matriz en la cpu
finale=mai.reshape((M,N))

#Imagen de salida para la transformada de fourier
mostrar((np.log10(np.abs(finale))),"fft a la Transformada de Fourier")

#Se redefine la dimensión del mallado
grid_dim = (M // (block_dim[0]), N// (block_dim[1]),1)

#Se inicia la función para el último correjimiento de fase
Fresnel2(final,dest_gpu,dest_copia, m_gpu, n_gpu, np.float32(dx_), np.float32(dy_),np.float32(lamb),np.float32(z),np.float32(pi), np.int32(N),np.int32(M), np.float32(dx), np.float32(dy), block=(16,16,1), grid=grid_dim)

#La variable mai es un nombre temporal de recuperación
mai=final.get()

#Se recrea la matriz en la cpu
finale=mai.reshape((M,N))

#Imagen de salida al multiplicar por el último correjimiento de fase
mostrar((np.abs(finale)),"Imagen final")