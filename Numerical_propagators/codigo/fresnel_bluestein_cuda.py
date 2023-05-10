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
z = 0.4
pi = mt.pi
# diferencial en el caso de la imagen final

dx_ = 9e-5
dy_ = 9e-5

# Definición de matrices de relleno.

padx = int(N/2)
pady = int(M/2)

# Parametros para cuda
n = n.astype(np.float32)
m = m.astype(np.float32)
U = U.astype(np.float32)

U_gpu = gpuarray.to_gpu(U)
m_gpu = gpuarray.to_gpu(m)
n_gpu = gpuarray.to_gpu(n)


mod = SourceModule("""
#include <math.h>
#include <stdio.h>
#include <cuComplex.h>

__global__ void Fase1(float *U_gpu,cuComplex *dest_copia, cuComplex *dest_gpu, float *m_gpu, float *n_gpu, float dx, 
                        float dy, float dx_, float dy_, float lamb, float z, float pi, int N, int M)
{
    int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
    int i2= ((col*M)+(fila));

    dest_copia[i2].x=cos( pi*((dx*(dx-dx_)*pow(n_gpu[i2],2))+(dy*(dy-dy_)*pow(m_gpu[i2],2)))/(lamb*z));
    dest_copia[i2].y=sin( pi*((dx*(dx-dx_)*pow(n_gpu[i2],2))+(dy*(dy-dy_)*pow(m_gpu[i2],2)))/(lamb*z));

	dest_gpu[i2].x = dx*dy*U_gpu[i2]*dest_copia[i2].x;
    dest_gpu[i2].y = dx*dy*U_gpu[i2]*dest_copia[i2].y;
}
__global__ void multi(cuComplex *final,cuComplex *dest_copia, cuComplex *dest_gpu, int N)
{
    int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
    int i2= (fila*2*N+col);

    final[i2].x= ((dest_gpu[i2].x*dest_copia[i2].x)-(dest_gpu[i2].y*dest_copia[i2].y));
    final[i2].y= ((dest_gpu[i2].y*dest_copia[i2].x)+(dest_gpu[i2].x*dest_copia[i2].y));
}	
__global__ void Fase2(cuComplex *dest_copia, float *m_gpu, float *n_gpu, float dx, float dy, float dx_, float dy_, float lamb, float z, float pi, int N, int M)
{
    int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
    int i2= ((col*M)+(fila));

    dest_copia[i2].x=cos(pi*((dx*dx_*pow(n_gpu[i2],2))+(dy*dy_*pow(m_gpu[i2],2)))/(lamb*z));
    dest_copia[i2].y=sin(pi*((dx*dx_*pow(n_gpu[i2],2))+(dy*dy_*pow(m_gpu[i2],2)))/(lamb*z));
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
__global__ void fft_shift2(cuComplex *final,cuComplex *dest_gpu, int N, int M)
{
	//Descriptores de cada hilo

    int n2 = N;
	int m2 = M;

    int fila = (blockIdx.x*blockDim.x + threadIdx.x);
	int col = (blockIdx.y*blockDim.y+ threadIdx.y);
	int fila2 = fila + n2;
	int col2 = col + m2;
    
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
__global__ void padding(cuComplex *small, cuComplex *big, int N, int M,int N1,int M1)
{
    int fila = blockIdx.x*blockDim.x+threadIdx.x;
	int col  = blockIdx.y*blockDim.y+threadIdx.y;
    int col2 = col+N1;
    int fila2 = fila+M1;
    big[((fila2*2*N))+(col2)].x = small[(fila*N)+col].x;
    big[((fila2*2*N))+(col2)].y = small[(fila*N)+col].y;
    
}
__global__ void padding_inv(cuComplex *small, cuComplex *big, int N, int M,int N1,int M1)
{
    int fila = blockIdx.x*blockDim.x+threadIdx.x;
	int col  = blockIdx.y*blockDim.y+threadIdx.y;
    int col2 = col+M1;
    int fila2 = fila+N1;
    small[(fila*N)+col].x = big[((fila2*2*N))+(col2)].x;
    small[(fila*N)+col].y = big[((fila2*2*N))+(col2)].y;
    
}
__global__ void fase3(cuComplex *final,cuComplex *dest_copia, cuComplex *dest_gpu, float *m_gpu, float *n_gpu,  float dx_, float dy_, float lamb, float z, float pi, int N, int M,float dx, float dy)
{

    int fila = blockIdx.x*blockDim.x+threadIdx.x;
	int col  = blockIdx.y*blockDim.y+threadIdx.y;
    int i2= ((col*M)+(fila));

    dest_copia[i2].x = __cosf((2*pi*z/lamb)+(pi / (lamb * z) * (((dx_*(dx_-dx)*(pow(n_gpu[i2], 2))+ ((dy_*(dy_-dy))*pow(m_gpu[i2], 2)))))));
    dest_copia[i2].y = __sinf((2*pi*z/lamb)+(pi / (lamb * z) * (((dx_*(dx_-dx)*(pow(n_gpu[i2], 2))+ ((dy_*(dy_-dy))*pow(m_gpu[i2], 2)))))));
    dest_gpu[i2].x = (-1)*((final[i2].y * dest_copia[i2].x) + (final[i2].x * dest_copia[i2].y))/(lamb*z);
    dest_gpu[i2].y = (-1)*((final[i2].x * dest_copia[i2].x) - (final[i2].y * dest_copia[i2].y))/(lamb*z);

}

""")
# Se llaman las funciones de la GPU para guardarlas en la CPU
Fase1 = mod.get_function("Fase1")
fft_shift = mod.get_function("fft_shift")
Fase2 = mod.get_function("Fase2")
multi = mod.get_function("multi")
fft_shift2 = mod.get_function("fft_shift")
fase3 = mod.get_function("fase3")
padding = mod.get_function("padding")
padding_inv = mod.get_function("padding_inv")

# Variables de uso a lo largo de las funciones
dest_copia = gpuarray.empty((N, M), np.complex64)
dest_gpu = gpuarray.empty((N, M), np.complex64)
x_gpu = gpuarray.empty((2*N, 2*M), np.float32)
final = gpuarray.empty((N, M), np.complex64)
big = gpuarray.empty((2*N, 2*M), np.complex64)
big_shift = gpuarray.empty((2*N, 2*M), np.complex64)
big2 = gpuarray.empty((2*N, 2*M), np.complex64)
big_shift2 = gpuarray.empty((2*N, 2*M), np.complex64)

N1 = N/2
M1 = M/2
start = timer()
# start2 = timer()
# Con el grid_dim se puede ajustar la malla que se va a considerar
block_dim = (16, 16, 1)


# Mallado para la primera correción de fase
grid_dim = (M // (block_dim[0]), N // (block_dim[1]), 1)

# Se aplica la primera función
Fase1(U_gpu, dest_copia, dest_gpu, m_gpu, n_gpu, np.float32(dx), np.float32(dy), np.float32(dx_), np.float32(dy_), np.float32(
    lamb), np.float32(z), np.float32(pi), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

# Se llama mai como una variable de transporte de datos de la GPU a la CPU
# mai = dest_gpu.get()

# Se dimensiona la variable mai
# finale = mai.reshape((M, N))

# Imagen de salida
# mostrar((np.abs(finale)), "Eso",
#        "pixeles en el eje x", "pixeles en el eje y")

# Segunda fase
Fase2(final, m_gpu, n_gpu, np.float32(dx), np.float32(dy), np.float32(dx_), np.float32(dy_), np.float32(
    lamb), np.float32(z), np.float32(pi), np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
# mai = final.get()

# Se dimensiona la variable mai
# finale = mai.reshape((M, N))

# Imagen de salida
# mostrar((np.abs(finale)), "Fase 2",
#        "pixeles en el eje x", "pixeles en el eje y")
grid_dim = (M // (block_dim[0]), N // (block_dim[1]), 1)
padding(dest_gpu, big, np.int32(N), np.int32(M), np.int32(N1),
        np.int32(M1), block=block_dim, grid=grid_dim)
padding(final, big2, np.int32(N), np.int32(M), np.int32(N1),
        np.int32(M1), block=block_dim, grid=grid_dim)
# mai = big.get()
# Se dimensiona la variable mai
# finale = mai.reshape((2*M, 2*N))

# Imagen de salida

# mostrar((np.abs(finale)), "Prueba de ",
#        "pixeles en el eje x", "pixeles en el eje y")
# mai = big2.get()

# Se dimensiona la variable mai
# finale = mai.reshape((2*M, 2*N))

# Imagen de salida
# mostrar(np.log(np.abs(finale)), "Padding de fase2",
#       "pixeles en el eje x", "pixeles en el eje y")

fft_shift2(big_shift, big, np.int32(2*N), np.int32(
    2*M), block=block_dim, grid=grid_dim)

# mai = big_shift.get()

# Se dimensiona la variable mai
# finale = mai.reshape((2*M, 2*N))

# Imagen de salida
# mostrar((np.abs(finale)), "fft_shift a la fase2",
#        "pixeles en el eje x", "pixeles en el eje y")
# primera fft (para la fase 1)
plan = cu_fft.Plan((2*N, 2*M), np.complex64, np.complex64)
cu_fft.fft(big_shift, big, plan)
# mai = big.get()

# Se dimensiona la variable mai
# finale = mai.reshape((2*M, 2*N))

# Imagen de salida
# mostrar((np.abs(finale)), "Primer FFT",
#        "pixeles en el eje x", "pixeles en el eje y")

fft_shift2(big_shift, big, np.int32(
    2*N), np.int32(2*M), block=block_dim, grid=grid_dim)

# mai = big_shift.get()

# Se dimensiona la variable mai
# finale = mai.reshape((2*M, 2*N))

# Imagen de salida

# mostrar((np.abs(finale)), "shift posterior a fft 2",
#        "pixeles en el eje x", "pixeles en el eje y")

# Segunda fft (para la fase 2)
fft_shift2(big_shift2, big2, np.int32(
    2*N), np.int32(2*M), block=block_dim, grid=grid_dim)

# mai = big_shift2.get()

# Se dimensiona la variable mai
# finale = mai.reshape((2*M, 2*N))

# Imagen de salida

# mostrar((np.abs(finale)), "fft shift del segundo FFT",
#        "pixeles en el eje x", "pixeles en el eje y")

cu_fft.fft(big_shift2, big2, plan)

mai = big2.get()

# Se dimensiona la variable mai
finale = mai.reshape((2*M, 2*N))

# Imagen de salida
# mostrar((np.abs(finale)), "fft a la fase2",
#        "pixeles en el eje x", "pixeles en el eje y")


fft_shift2(big_shift2, big2, np.int32(
    2*N), np.int32(2*M), block=block_dim, grid=grid_dim)

# mai = big_shift2.get()

# Se dimensiona la variable mai
# finale = mai.reshape((2*M, 2*N))

# Imagen de salida

# mostrar((np.abs(finale)), "shift posterior a fft 2",
#        "pixeles en el eje x", "pixeles en el eje y")

grid_dim = (2*M // (block_dim[0]), 2*N // (block_dim[1]), 1)

multi(big, big_shift2, big_shift, np.int32(
    M), block=block_dim, grid=grid_dim)
# mai = big_shift.get()

# Se dimensiona la variable mai
# finale = mai.reshape((2*M, 2*N))

# Imagen de salida
# mostrar((np.abs(finale)), "fases en el dominio de la frecuencia",
#        "pixeles en el eje x", "pixeles en el eje y")

grid_dim = (M // (block_dim[0]), N // (block_dim[1]), 1)
fft_shift2(big_shift, big, np.int32(
    2*N), np.int32(2*M), block=block_dim, grid=grid_dim)

# mai = big_shift.get()

# Se dimensiona la variable mai
# finale = mai.reshape((2*M, 2*N))

# mostrar((np.abs(finale)), "fft shift previo a ifft",
#        "pixeles en el eje x", "pixeles en el eje y")
cu_fft.ifft(big_shift, big, plan)

# mai = big.get()

# Se dimensiona la variable mai
# finale = mai.reshape((2*M, 2*N))

# mostrar((np.abs(finale)), "ifft",
#        "pixeles en el eje x", "pixeles en el eje y")

fft_shift2(big_shift, big, np.int32(
    2*N), np.int32(2*M), block=block_dim, grid=grid_dim)

# mai = final.get()

# Se dimensiona la variable mai
# finale = mai.reshape((M, N))

# Imagen de salida
# mostrar(np.log(np.abs(finale)), "final",
# "pixeles en el eje x", "pixeles en el eje y")

padding_inv(dest_gpu, big_shift, np.int32(N), np.int32(M), np.int32(N1),
            np.int32(M1), block=block_dim, grid=grid_dim)

grid_dim = (M // (block_dim[0]), N // (block_dim[1]), 1)


# Se dimensiona la variable mai

# Imagen de salida
# mai = dest_gpu.get()
# finale = mai.reshape((M, N))
# mostrar((np.abs(finale)), "antes de la fase 3",
#        "pixeles en el eje x", "pixeles en el eje y")

fase3(dest_gpu, dest_copia, final, m_gpu, n_gpu, np.float32(dx_), np.float32(dy_), np.float32(
    lamb), np.float32(z), np.float32(pi), np.int32(N), np.int32(M), np.float32(dx), np.float32(dy), block=block_dim, grid=grid_dim)

mai = final.get()
finale = mai.reshape((M, N))
# mostrar(np.log(np.abs(finale)), "final",
#        "pixeles en el eje x", "pixeles en el eje y")
print("without GPU:", timer()-start)
# print("without GPU:", timer()-start2)
dual_img(U, np.log(np.abs(finale)), "Fresnel Bluestein")
