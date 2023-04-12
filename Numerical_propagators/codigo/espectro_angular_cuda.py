import numpy as np
from numpy import asarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from calculo_imagenes import *
from timeit import default_timer as timer
# importa una imagen y la replica en formato png para poder jugar con ella
archivo = "+3cm.tif"
replica = lectura(archivo)
U = asarray(replica)
width, height = replica.size

# Longitud de las imagenes en la coordenada respectiva [m]


# N es el número de píxeles en el eje x de la imagen origen

N = height

# M es el número de pixeles en el eje y de la imagen de origen

M = width

# Definición vectorial de los dx y dy
x = np.arange(0, N, 1)
y = np.arange(0, M, 1)

# P es el numero de píxeles en el eje x de A
# Q es el numero de píxeles en el eje y de A
n, m = np.meshgrid(x - (N/2), y - (M/2))


# calculo de delta x y y
dx = 6.9
fx = 1/(dx*N)
dy = 6.9
fy = 1/(dy*M)

# distancia en la que se posiciona la imagen [m]

z = 70000

# Numéro y longitud de onda

lamb = 0.633
k = (2*np.pi/lamb)
U = np.float32(U)
n = n.astype(np.float32)
m = m.astype(np.float32)
U = U.astype(np.float32)
x = U.astype(np.float32)
U_gpu = gpuarray.to_gpu(U)
x_gpu = gpuarray.to_gpu(U)
n_gpu = gpuarray.to_gpu(n)
m_gpu = gpuarray.to_gpu(m)

# Declaración del kernel
mod = SourceModule("""
#include <math.h>
#include <stdio.h>
#include <cuComplex.h>
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
__global__ void fft_shift_img(cuComplex *final,float *U_gpu, int N, int M)
{
	int n2 = N / 2;
	int m2 = M / 2;

	//Descriptores de cada hilo
    int fila = blockIdx.x*blockDim.x+ threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	int fila2 = blockIdx.x*blockDim.x + threadIdx.x + m2;
	int col2 = blockIdx.y*blockDim.y + threadIdx.y + n2;
    final[fila*N+col].x = U_gpu[((fila2*N) + (col2))];
    final[fila*N+col].y = 0;
    final[fila2*N+col2].x = U_gpu[(fila*N + col)];
    final[fila2*N+col2].y = 0;
    final[fila*N+col2].x = U_gpu[(fila2*N + col)];
    final[fila*N+col2].y = 0;
    final[fila2*N+col].x = U_gpu[(fila*N + col2)];
    final[fila2*N+col].y = 0;
}
    
__global__ void fase(cuComplex *final,cuComplex *dest_copia, cuComplex *dest_gpu, float *m_gpu, float *n_gpu, float fx, float fy, float z,float pi, float N, float lamb)
{

    int fila = blockIdx.x*blockDim.x+threadIdx.x;
	int col = blockIdx.y*blockDim.y+threadIdx.y;
    int i2= ((fila*N)+col);

    dest_copia[i2].x = __cosf(2*pi*z*pow(pow(lamb,2)-(pow(fx*n_gpu[i2],2)+pow(fy*m_gpu[i2],2)),0.5));
    dest_copia[i2].y = __sinf(2*pi*z*pow(pow(lamb,2)-(pow(fx*n_gpu[i2],2)+pow(fy*m_gpu[i2],2)),0.5));
    dest_gpu[i2].y = ((final[i2].y * dest_copia[i2].x) + (final[i2].x * dest_copia[i2].y));
    dest_gpu[i2].x = ((final[i2].x * dest_copia[i2].x) - (final[i2].y * dest_copia[i2].y));

}
""")
fft_shift_U = mod.get_function("fft_shift_img")
fft_shift = mod.get_function("fft_shift")
fase = mod.get_function("fase")
dest_copia = gpuarray.empty((N, M), np.complex64)
dest_gpu = gpuarray.empty((N, M), np.complex64)
x_gpu = gpuarray.empty((N, M), np.complex64)
final = gpuarray.empty((N, M), np.complex64)

block_dim = (16, 16, 1)

# Mallado para la primera correción de fase
grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)

fft_shift_U(x_gpu, U_gpu, np.int32(M), np.int32(N),
            block=block_dim, grid=grid_dim)
plan = cu_fft.Plan(U.shape, np.complex64, np.complex64)
mai = x_gpu.get()

# Se recrea la matriz de salida de el shift en la cpu
finale = mai.reshape((N, M))
mostrar(((np.abs(finale))), "fft_shift U",
        "pixeles en el eje x", "pixeles en el eje y")
cu_fft.fft(x_gpu, dest_gpu, plan)
mai = dest_gpu.get()
fft_shift(final, dest_gpu, np.int32(M), np.int32(
    N), block=block_dim, grid=grid_dim)

grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)

# Se recrea la matriz de salida de el shift en la cpu
finale = mai.reshape((N, M))

fase(final, dest_copia, dest_gpu, m_gpu, n_gpu, np.int32(fx), np.int32(
    fy), np.int32(np.pi), np.int32(z), np.int32(lamb), block=block_dim, grid=grid_dim)
mai = dest_gpu.get()
finale = mai.reshape((N, M))


grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)
fft_shift(final, dest_gpu, np.int32(M), np.int32(
    N), block=block_dim, grid=grid_dim)

cu_fft.ifft(dest_gpu, final, plan)
fft_shift(dest_gpu, final, np.int32(M), np.int32(
    N), block=block_dim, grid=grid_dim)

mai = dest_gpu.get()
finale = mai.reshape((N, M))
mostrar(((np.abs(finale))), "final",
        "pixeles en el eje x", "pixeles en el eje y")
dual_img(U, np.abs(finale), "Espectro angular")
