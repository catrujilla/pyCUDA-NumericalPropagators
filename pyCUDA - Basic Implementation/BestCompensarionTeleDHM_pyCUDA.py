#Best phase compensation GPU-accelerated using other library (pyCUDA)

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import skcuda.fft as cu_fft

import imageio
import struct

import math

from blessed import Terminal

import numpy as np

from timeit import default_timer as timer

from PIL import Image
import matplotlib.pyplot as plt

mod = SourceModule("""
  #include <cuComplex.h>
  #include <math_functions.h>
    
  __global__ void fft_shift_complex(cuComplex* __restrict__ arregloC, float *__restrict__ d_temp13x, int width, int height)
  {

	int m2 = width / 2;
	int n2 = height / 2;

	//Descriptores de cada hilo
    int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	int fila2 = blockIdx.x*blockDim.x + threadIdx.x + m2;
	int col2 = blockIdx.y*blockDim.y + threadIdx.y + n2;
    
    d_temp13x[fila*width + col] = arregloC[fila*width + col].x;  //Guardo el primer cuadrante
	arregloC[fila*width + col].x = arregloC[fila2*width + col2].x;  //en el primer cuadrante estoy poniendo lo que hay en el tercero
	arregloC[fila2*width + col2].x = d_temp13x[fila*width + col];//En el tercer cuadrante estoy poniendo lo que habia en el primero

	d_temp13x[fila*width + col] = arregloC[fila*width + col].y;  //Lo mismo anterior pero para los imaginarios
	arregloC[fila*width + col].y = arregloC[fila2*width + col2].y;
	arregloC[fila2*width + col2].y = d_temp13x[fila*width + col];

	d_temp13x[fila*width + col] = arregloC[fila*width + col2].x;//Guardo Cuadrante dos
	arregloC[fila*width + col2].x = arregloC[fila2*width + col].x;  //En el segundo guardo lo que hay en el cuarto
	arregloC[fila2*width + col].x = d_temp13x[fila*width + col];//En el cuarto guardo lo que estaba en el segundo

	d_temp13x[fila*width + col] = arregloC[fila*width + col2].y; //Lo mismo que en el anterior
	arregloC[fila*width + col2].y = arregloC[fila2*width + col].y;
	arregloC[fila2*width + col].y = d_temp13x[fila*width + col];
  }
  
  __global__ void CambioTipoVariableUnaMatrix(float *__restrict__ real, cuComplex* __restrict__ arregloC, int width, int height)
  {

    //Descriptores de cada hilo
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int fila = blockIdx.y*blockDim.y + threadIdx.y;

    arregloC[fila*width + col].x = real[fila*width + col];
    arregloC[fila*width + col].y = 0.0;
  } 
  
  __global__ void modulo(cuComplex *__restrict__ arregloC, float *temp_intensidad, float *other, int quadrant, int width, int height)
  {
  //Descriptores de cada hilo
  
  int fila = blockIdx.x*blockDim.x + threadIdx.x;
  int col = blockIdx.y*blockDim.y + threadIdx.y;
    
  int Espacio = int(height*0.1);
    
  //Intensity computing considering quadrant of point of maxima
  other[fila*width + col] = fma(arregloC[fila*width + col].x, arregloC[fila*width + col].x, arregloC[fila*width + col].y * arregloC[fila*width + col].y);
  temp_intensidad[fila*width + col] = 1.414; //The modulo of 1 + 1j

  temp_intensidad[fila*width + col] = (quadrant == 1 && fila < (height >> 1) - Espacio && col < (width >> 1) - Espacio) ? other[fila*width + col] : temp_intensidad[fila*width + col];
  temp_intensidad[fila*width + col] = (quadrant == 2 && col > (height >> 1) + Espacio && fila < (width >> 1) - Espacio) ? other[fila*width + col] : temp_intensidad[fila*width + col];
  temp_intensidad[fila*width + col] = (quadrant == 3 && fila > (height >> 1) + Espacio && col < (width >> 1) - Espacio) ? other[fila*width + col] : temp_intensidad[fila*width + col];
  temp_intensidad[fila*width + col] = (quadrant == 4 && col > (height >> 1) + Espacio && fila > (width >> 1) + Espacio) ? other[fila*width + col] : temp_intensidad[fila*width + col];
  }
   
  __global__ void CircularROI_Preparation( cuComplex *__restrict__ arregloC, int r, int fx_max, int fy_max, int width, int height)
  {
	//Thread descriptors
	int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	
	//Only points inside the circular ROI are going to be tested
	if (sqrt( (float)((fila - fx_max) * (fila - fx_max) + (col - fy_max) * (col - fy_max) )) < r) {
        arregloC[fila*width + col].x = arregloC[fila*width + col].x;
        arregloC[fila*width + col].y = arregloC[fila*width + col].y;
	} else {

        arregloC[fila*width + col].x = 1;
        arregloC[fila*width + col].y = 1;
	}

  }
  
  #define BLOCK_SIZE 128

__global__ void Compensacion_SalidaFase(float *__restrict__ odata_real, float *__restrict__ odata_imag, cuComplex *__restrict__ arregloC,
  cuComplex *__restrict__ arregloTemp, float theta_x, float theta_y, float k, float dx, float dy, int width, int height)
{
	//Descriptores de cada hilo    
    int fila = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	
	//real and imaginary parts of the tilted reference
	float ref_real = __cosf( k * (__sinf(theta_x) * (fila-(width/2)) * dx + __sinf(theta_y) * (col-(height/2)) * dy));

	float ref_imag = __sinf( k * (__sinf(theta_x) * (fila-(width/2)) * dx + __sinf(theta_y) * (col-(height/2)) * dy));

    //Pointwise multiplication of complex "arrays""
	odata_real[col*width + fila] = (arregloC[col*width + fila].x * ref_real) - (arregloC[col*width + fila].y * ref_imag);
	odata_imag[col*width + fila] = (arregloC[col*width + fila].y * ref_real) + (arregloC[col*width + fila].x * ref_imag);

    odata_real[col*width + fila] = atan2( odata_imag[col*width + fila], odata_real[col*width + fila] );
    arregloTemp[col*width + fila].x = ref_real;
    arregloTemp[col*width + fila].y = ref_imag;
}



  __global__ void getStats(float *__restrict__ pArray, float *__restrict__ pMaxResults, float *__restrict__ pMinResults)
  {
	// Declare arrays to be in shared memory.
	// 256 elements * (4 bytes / element) * 2 = 2KB.
	__shared__ float min[256];
	__shared__ float max[256];

	// Calculate which element this thread reads from memory
	int arrayIndex = 256 * 128 * blockIdx.y + 256 * blockIdx.x + threadIdx.x;
	min[threadIdx.x] = max[threadIdx.x] = pArray[arrayIndex];
	__syncthreads();


	int nTotalThreads = blockDim.x;	// Total number of active threads

	while (nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < halfPoint)
		{
			// Get the shared value stored by another thread
			float temp = min[threadIdx.x + halfPoint];
			if (temp < min[threadIdx.x]) min[threadIdx.x] = temp;
			temp = max[threadIdx.x + halfPoint];
			if (temp > max[threadIdx.x]) max[threadIdx.x] = temp;
		}


		nTotalThreads = (nTotalThreads >> 1);	// divide by two.
	}

	// At this point in time, thread zero has the min, max, and average
	// It's time for thread zero to write it's final results.
	// Note that the address structure of pResults is different, because
	// there is only one value for every thread block.

	if (threadIdx.x == 0)
	{
		pMaxResults[128 * blockIdx.y + blockIdx.x] = max[0];
		pMinResults[128 * blockIdx.y + blockIdx.x] = min[0];

	}
  }
  
  __global__ void escalamiento(float *__restrict__ temp, int width, int height, float maximo, float minimo)
  {

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	temp[fila*width + col] = (temp[fila*width + col]) - minimo;
	temp[fila*width + col] = (temp[fila*width + col]) / (maximo - minimo);
	temp[fila*width + col] = (temp[fila*width + col]) * 255;
	//Ac? tenemos todas los pixeles escalados a 8 bits (255 niveles de gris)

  }
  
  __global__ void Umbralizacion( float *__restrict__ temp, int umbral, int width, int height) 
  {
	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	//Calculo de la intensidad
	if (temp[fila*width + col] >= umbral) {
		temp[fila*width + col] = 1;
	}
	else {
		temp[fila*width + col] = 0;
	}

  }

  __global__ void fft_inverse_correction(cuComplex *__restrict__ arregloC, int width, int height) 
  {
	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

    arregloC[fila*width + col].x = arregloC[fila*width + col].x;
    arregloC[fila*width + col].y = (-1)*arregloC[fila*width + col].y;

  }

  
  __global__ void Sumatoria(float *__restrict__ pArray, float *__restrict__ pDesviacion) 
  {
	// Declare arrays to be in shared memory.
	// 128 elements * (4 bytes / element) * 2 = 2KB.
	__shared__ float avg[256];

	// Calculate which element this thread reads from memory
	int arrayIndex = 256 * 128 * blockIdx.y + 256 * blockIdx.x + threadIdx.x;
	avg[threadIdx.x] = pArray[arrayIndex];
	__syncthreads();


	int nTotalThreads = blockDim.x;	// Total number of active threads

	while (nTotalThreads > 1)
	{
		int halfPoint = (nTotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < halfPoint)
		{
			// when calculating the average, sum and divide
			avg[threadIdx.x] += avg[threadIdx.x + halfPoint];
			//avg[threadIdx.x] /= 2;
		}

		nTotalThreads = (nTotalThreads >> 1);	// divide by two.
	}

	if (threadIdx.x == 0)
	{
		pDesviacion[128 * blockIdx.y + blockIdx.x] = avg[0];

	}

  }


  """)
  
print ("Starting memory allocation and CUDA context initialization")
start = timer()	#Start to count time

#Best off-axis phase compensation method

# Read the image file
#image = imageio.imread('Holo_tele_RBCs_40x.jpg')
image = imageio.imread('fly 20x.jpg')
#image = imageio.imread('+3cm.jpg')

# Convert the image to a NumPy array
image_array = np.array(image)

# To access a specific component of a NumPy array
hologram = image_array[:, :, 0]

#Parameters for the reconstruction (Thorlabs cam, image plane hologram z = 0):
z = 0.00
dx = 2.4 #6.9 for the USAF test, 5.2 for the RBCs, 2.4 for the fly
dy = 2.4 #6.9 for the USAF test
lambd = 0.532 #0.532 for the fly, 0.633 for the USAF and RBCs
pi = 3.141592
k = 2 * pi / lambd

#This variable determines in which quadrant is located the real image in the POWER spectrum. 
#If this variable is assigned to 0, then no filter is applied.
quadrant = 2 #quadrant to search for the point fo maxima

#CompensationWave search parameters
s = 2
step = 10

#Image data variables
# Get the shape of the array
shape = hologram.shape
M = shape[0]
N = shape[1]
numElements = N * M

#Reduction process variables (search of the maximum and minimum)
THREADS_PER_BLOCK = int(256)
BLOCKS_PER_GRID_ROW = int(128)

#Allocate and fill the host input data
real = np.zeros(numElements, dtype=np.float32)

temp = np.zeros(numElements, dtype=np.float32)
cpu_bufferFFT2 = np.zeros((M,N), dtype=np.complex64)

#h_resultMax[] = new float[numElements / THREADS_PER_BLOCK * Sizeof.FLOAT];
h_resultMax = np.zeros(int(numElements / THREADS_PER_BLOCK * struct.calcsize('f')), dtype=np.float32)
#h_resultMax = np.zeros(int(numElements / THREADS_PER_BLOCK * 4), dtype=np.float32)
h_resultMin = np.zeros(int(numElements / THREADS_PER_BLOCK * struct.calcsize('f')), dtype=np.float32)
#h_resultMin = np.zeros(int(numElements / THREADS_PER_BLOCK * 4), dtype=np.float32)
h_resultDes = np.zeros(int(numElements / THREADS_PER_BLOCK * struct.calcsize('f')), dtype=np.float32)

#Loading of the hologram data in arrays to initiate the process
#(if the subtraction of the reference is needed, this is the place to do it...)
for x in range(M):
    for y in range(N):
        real[x * M + y] = hologram [x, y]


# Allocate the device input data, and copy the
# host input data to the device
devicereal = gpuarray.to_gpu(real)# From numpy array to GPUarray

#Allocate device temp memory
devicetemp = cuda.mem_alloc(temp.nbytes)
cuda.memcpy_htod(devicetemp, temp)

#Allocate device memory for the reductions
d_resultMax = cuda.mem_alloc(h_resultMax.nbytes)
cuda.memcpy_htod(d_resultMax, h_resultMax)
d_resultMin = cuda.mem_alloc(h_resultMin.nbytes)
cuda.memcpy_htod(d_resultMin, h_resultMin)
d_resultDes = cuda.mem_alloc(h_resultDes.nbytes)
cuda.memcpy_htod(d_resultDes, h_resultDes)

# Initialise output GPUarray 
bufferFFT2 = gpuarray.empty((M,N), np.complex64)
bufferFFT1 = gpuarray.zeros((M,N), np.complex64)

#Variables to define the size of each block, thus, the number of blocks in each dimension of the grid. 
#The latter based on the number of threads and the size oft he array (image)
block_size_x = 16
block_size_y = 16

blockGridWidth = int(BLOCKS_PER_GRID_ROW)
blockGridHeight = int((numElements / THREADS_PER_BLOCK) / blockGridWidth)

block_dim = (block_size_x, block_size_y,1)

print("Processing time:", timer()-start) #Time for fft_shift execution

print ("Starting process")
start = timer()	#Start to count time

#let's put the data in a cuComplex-type array
grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
CambioTipoVariableUnaMatrix = mod.get_function("CambioTipoVariableUnaMatrix")
CambioTipoVariableUnaMatrix(devicereal, bufferFFT1, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()

grid_dim = (M // (2*block_dim[0]), N // (2*block_dim[1]),1)
fft_shift2 = mod.get_function("fft_shift_complex")
fft_shift2(bufferFFT1, devicetemp, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()

# Forward FFT
plan_forward = cu_fft.Plan(bufferFFT1.shape, np.complex64, np.complex64)
cu_fft.fft(bufferFFT1, bufferFFT1, plan_forward)
pycuda.driver.Context.synchronize()

grid_dim = (M // (2*block_dim[0]), N // (2*block_dim[1]),1)
fft_shift2 = mod.get_function("fft_shift_complex")
fft_shift2(bufferFFT1, devicetemp, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()


'''
print ("Fourier spectrum of hologram")
cpu_bufferFFT2 = bufferFFT1.get()

amp = np.log10(np.abs(cpu_bufferFFT2))
max = np.max(amp)
min = np.min(amp)
amp = 255*(amp - min)/(max - min)

plt.imshow(amp, cmap='gray')
plt.show()
'''



#In this part we search for the best ROI (based on the x% points of maxima of the circular ROI)
#These values can be determined by means of the user (looking at the power spectrum) 
#or by executing an automatic search of the point of maxima.
#Automatic method:

#This kernel computes the module of the array 'bufferFFT1' in 'devicetemp' and 
#then applies the mask over the selected quadrant leaving the result in 'devicetemp'
grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
modulo = mod.get_function("modulo")
modulo(bufferFFT1, devicetemp, devicereal, np.int32(quadrant), np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()



'''
print ("Fourier spectrum of hologram with mask")
cuda.memcpy_dtoh(temp, devicetemp)

temp = np.log10(temp)

for x in range(M):
    for y in range(N):
        hologram[x, y] =  temp[x * M + y].astype(np.uint8)

amp = hologram
plt.imshow(amp, cmap='gray')
plt.show()
'''


# Find the index of the maximum value
cuda.memcpy_dtoh(temp, devicetemp) #First, let's transfer it from D to H
# Find the index of the maximum value along the first axis (since CUDA works with vectors, not matrices)
col_max_index = temp.argmax(axis=0)

fx_max = int(col_max_index/M) #Location in fx
fy_max = col_max_index%M #location in fy

#print (fx_max, fy_max)

d = math.sqrt((fx_max - N / 2) * (fx_max - N / 2) + (fy_max - N / 2) * (fy_max - N / 2));
#print (d) #Distance between DC term and first difractin order

r = d / 3 #radius of the +1 difraction order in difraction-limited conditions

#Correction of the +1 D.O size regarding ita center location 
if fx_max - r < 0:
    r = fx_max - 1
    print("Circular ROI radius (corrected):", r)  
if fy_max - r < 0:
    r = fy_max - 1
    print("Circular ROI radius (corrected):", r)  
if fx_max + r > N:
    r = N - fx_max - 1
    print("Circular ROI radius (corrected):", r)  
if fy_max + r > N:
    r = N - fy_max - 1
    print("Circular ROI radius (corrected):", r)  

#Let's apply the spatial filtering with a circular ROI of radius 'r' and centered at (fx, fy)
grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
CircularROI_Preparation = mod.get_function("CircularROI_Preparation")
CircularROI_Preparation(bufferFFT1, np.int32(r), np.int32(fx_max), np.int32(fy_max), np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()


'''
print ("After Circular ROI")
cpu_bufferFFT2 = bufferFFT1.get()
amp = np.log10((np.abs(cpu_bufferFFT2)))

max = np.max(amp)
min = np.min(amp)
print (max, min)
amp = 255*(amp - min)/(max - min)

plt.imshow(amp, cmap='gray')
plt.show()
'''


#Let's come back to spatial domain
#IFFT
#Let's correct the output of this foward fft (we need the inverse fft)
grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
fft_inverse_correction = mod.get_function("fft_inverse_correction")
fft_inverse_correction(bufferFFT1, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()

grid_dim = (M // (2*block_dim[0]), N // (2*block_dim[1]),1)
fft_shift3 = mod.get_function("fft_shift_complex")
fft_shift3(bufferFFT1, devicetemp, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()

cu_fft.fft(bufferFFT1, bufferFFT1, plan_forward)
pycuda.driver.Context.synchronize()

grid_dim = (M // (2*block_dim[0]), N // (2*block_dim[1]),1)
fft_shift3 = mod.get_function("fft_shift_complex")
fft_shift3(bufferFFT1, devicetemp, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()

#Let's correct the output of this foward fft (we need the inverse fft)
grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
fft_inverse_correction = mod.get_function("fft_inverse_correction")
fft_inverse_correction(bufferFFT1, np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()


'''
print ("spatial domain")
cpu_bufferFFT2 = bufferFFT1.get()
amp = np.abs(cpu_bufferFFT2)

max = np.max(amp)
min = np.min(amp)
amp = 255*(amp - min)/(max - min)

plt.imshow(amp, cmap='gray')
plt.show()
'''




# Phase Compensation operation

theta_x = 0.0
theta_y = 0.0

fx_0 = M / 2
fy_0 = N / 2

suma_maxima = 0  # small number for the metric (thresholding)
i_ROI = 0
i_out = 0
x_max_out = 0
y_max_out = 0

t = Terminal()

tmp = fy_max
fy_max = fx_max
fx_max = tmp

arrayX = np.linspace(fx_max - s, fx_max + s, step)
arrayY = np.linspace(fy_max - s, fy_max + s, step)

for fx_tmp in arrayX:
    for fy_tmp in arrayY:
        
        i_ROI = i_ROI + 1  # To identify the correct position of the ROI center
        #print (fx_tmp, fy_tmp)
        #print(t.clear_eol + str(fx_tmp), fy_tmp, end="\r")
        
        theta_x = math.asin((fx_0 - fx_tmp) * lambd / (M * dx))
        theta_y = math.asin((fy_0 - fy_tmp) * lambd / (N * dy))
        
        # Phase Compensation with specific theta_x and theta_y
        grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
        Compensacion_SalidaFase = mod.get_function("Compensacion_SalidaFase")
        Compensacion_SalidaFase(devicereal, devicetemp, bufferFFT1, bufferFFT2, np.float32(theta_x), np.float32(theta_y), np.float32(k), np.float32(dx), np.float32(dy), np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
        pycuda.driver.Context.synchronize()
        
        # Scaling
        grid_dim = (blockGridWidth, blockGridHeight, 1)
        block_dim_red = (THREADS_PER_BLOCK, 1, 1)
        getStats = mod.get_function("getStats")
        getStats(devicereal, d_resultMax, d_resultMin, grid = grid_dim, block=block_dim_red)
        pycuda.driver.Context.synchronize()
        
        # Copy the data back to the host
        cuda.memcpy_dtoh(h_resultMax, d_resultMax)
        cuda.memcpy_dtoh(h_resultMin, d_resultMin)
        #Each block returned one result, so lets finish this off with the cpu.
        #By using CUDA, we basically reduced how much the CPU would have to work by about 256 times.

        minimo = h_resultMin[0]
        maximo = h_resultMax[0]
        for i in range(1, numElements // THREADS_PER_BLOCK):
            if h_resultMin[i] < minimo:
                minimo = h_resultMin[i]
            if h_resultMax[i] > maximo:
                maximo = h_resultMax[i]

        # Now, the scaling
        grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
        escalamiento = mod.get_function("escalamiento")
        escalamiento(devicereal, np.int32(M), np.int32(N), np.float32(maximo), np.float32(minimo), grid = grid_dim, block=block_dim)
        pycuda.driver.Context.synchronize()
        
        
        
        '''
        temp = devicereal.get()
        for x in range(M):
            for y in range(N):
                hologram[x, y] = temp[x * M + y]
        amp = hologram
        plt.imshow(amp, cmap='gray')
        plt.show()
        '''
        
        
        
        
        

        # Thresholding
        threshold = 25
        grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
        Umbralizacion = mod.get_function("Umbralizacion")
        Umbralizacion(devicereal, np.int32(threshold), np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
        pycuda.driver.Context.synchronize()       
        

        # Summation
        grid_dim = (blockGridWidth, blockGridHeight, 1)
        Sumatoria = mod.get_function("Sumatoria")
        Sumatoria(devicereal, d_resultDes, grid = grid_dim, block=(THREADS_PER_BLOCK, 1, 1))

        # Copy the data back to the host
        cuda.memcpy_dtoh(h_resultDes, d_resultDes)

        aux_desviacion = h_resultDes[0]
        for i in range(1, (N * M) // THREADS_PER_BLOCK):
            aux_desviacion += h_resultDes[i]

        # Total summation
        sumatoria = aux_desviacion

        if sumatoria > suma_maxima:
            x_max_out = fx_tmp
            y_max_out = fy_tmp
            i_out = i_ROI
            suma_maxima = sumatoria

'''
print(
    "Point of the compensation wave for the best reconstruction:"
    + str(x_max_out)
    + ","
    + str(y_max_out)
    + "\n"
    + "Other info:"
    + str(i_out)
    + " and "
    + str(i_ROI)
    + "\n"
)  # To see what's going on

print("Metric value:" + str(suma_maxima) + "\n")  # To see what's going on
'''

# Calculating the best reconstruction

# Calculating the angle of the compensation wave
theta_x = math.asin((fx_0 - x_max_out) * lambd / (M * dx))
theta_y = math.asin((fy_0 - y_max_out) * lambd / (N * dy))

grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
Compensacion_SalidaFase2 = mod.get_function("Compensacion_SalidaFase")
Compensacion_SalidaFase2(devicereal, devicetemp, bufferFFT1, bufferFFT2, np.float32(theta_x), np.float32(theta_y), np.float32(k), np.float32(dx), np.float32(dy), np.int32(M), np.int32(N), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()




'''
# Let's see ref_wave built in device
print ("bulding text files")
cpu_bufferFFT2 = bufferFFT2.get()
ref_wave_d = cpu_bufferFFT2
#np.savetxt('ref_vave_d.txt', cpu_bufferFFT2, fmt='%.2f')


#let's build the ref wave for compensation in the host for comparison purposes
# Creating a mesh_grid to operate in world-coordinates
x = np.arange(0, N, 1)  # array x
y = np.arange(0, M, 1)  # array y
X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')  # meshgrid XY
ref_wave = np.exp(1j * k * ((math.sin(theta_x) * X * dx) + (math.sin(theta_y) * Y * dy)))
# Save the array to a text file
#np.savetxt('ref_vave_h.txt', ref_wave, fmt='%.2f')


cpu_bufferFFT2 = bufferFFT1.get()
holo_filtered = cpu_bufferFFT2
# Compensation of the tilting angle for the off-axis acquisition
reconstruction = holo_filtered * ref_wave
# module 2-pi phase retrieval
phase = np.angle(reconstruction)
# Thresholding process
minVal = np.amin(phase)
maxVal = np.amax(phase)
phase_sca = 255*(phase - minVal) / (maxVal - minVal)
plt.imshow(phase_sca, cmap='gray')
plt.show()
phase_sca = phase_sca.astype(np.uint8)
print("phase_sca file type: ", phase_sca.dtype)

# Convert the array to an image
image = Image.fromarray(phase_sca)
# Save the image to a file
image.save('image_h.jpg')

'''



# Scaling
grid_dim = (blockGridWidth, blockGridHeight, 1)
block_dim_red = (THREADS_PER_BLOCK, 1, 1)
getStats2 = mod.get_function("getStats")
getStats2(devicereal, d_resultMax, d_resultMin, grid = grid_dim, block=block_dim_red)
pycuda.driver.Context.synchronize()

# Copy the data back to the host
cuda.memcpy_dtoh(h_resultMax, d_resultMax)
cuda.memcpy_dtoh(h_resultMin, d_resultMin)
#Each block returned one result, so lets finish this off with the cpu.
#By using CUDA, we basically reduced how much the CPU would have to work by about 256 times.

minimo = h_resultMin[0]
maximo = h_resultMax[0]
for i in range(1, numElements // THREADS_PER_BLOCK):
    if h_resultMin[i] < minimo:
        minimo = h_resultMin[i]
    if h_resultMax[i] > maximo:
        maximo = h_resultMax[i]
        
#print (maximo,minimo)

# Now, the scaling
grid_dim = (M // (block_dim[0]), N // (block_dim[1]),1)
escalamiento2 = mod.get_function("escalamiento")
escalamiento2(devicereal, np.int32(M), np.int32(N), np.float32(maximo), np.float32(minimo), grid = grid_dim, block=block_dim)
pycuda.driver.Context.synchronize()





print("With GPU-processing:", timer()-start) #Time for fft_shift execution

temp = devicereal.get()
min = np.min(temp)
max = np.max(temp)
#print (min, max)

for x in range(M):
    for y in range(N):
        hologram[x, y] = temp[x * N + y]
        
amp = hologram
plt.imshow(amp, cmap='gray')
plt.show()

# Convert the array to an image
image = Image.fromarray(hologram)
# Save the image to a file
image.save('image_d.jpg')




'''
temp23 = bufferFFT1.get()
amp = np.log(np.abs(temp23)).astype(np.uint8)

plt.imshow(amp, cmap='gray')
plt.show()
'''

'''
cuda.memcpy_dtoh(temp, devicetemp)

for x in range(M):
    for y in range(N):
        hologram[x, y] =np.log(temp[y * N + x])
amp = hologram.astype(np.uint8)
plt.imshow(amp, cmap='gray')
plt.show()
'''


'''

left = bufferFFT2.get()
#amp = np.log(np.abs(left))

#plt.imshow(amp, cmap='gray')
#plt.show()



# To make the output array compatible with the numpy output
# we need to stack horizontally the y.get() array and its flipped version
# We must take care of handling even or odd sized array to get the correct 
# size of the final array   
if N//2 == N/2:
    right = np.roll(np.fliplr(np.flipud(bufferFFT2.get()))[:,1:-1],1,axis=0)
else:
    right = np.roll(np.fliplr(np.flipud(bufferFFT2.get()))[:,:-1],1,axis=0) 

#amp = np.log(np.abs(right))

#plt.imshow(amp, cmap='gray')
#plt.show()

# Get a numpy array back compatible with np.fft
cpu_bufferFFT2 = np.hstack((left,right))


#copy data back to the host memory
#cuda.memcpy_dtoh(cpu_bufferFFT2, bufferFFT2)
#cuda.memcpy_dtoh(imag, deviceimag)

amp = np.log(np.abs(cpu_bufferFFT2)).astype(np.uint8)

plt.imshow(amp, cmap='gray')
plt.show()

# Convert the array to an image
#image = Image.fromarray(amp)
# Save the image to a file
#image.save('image.jpg')

'''