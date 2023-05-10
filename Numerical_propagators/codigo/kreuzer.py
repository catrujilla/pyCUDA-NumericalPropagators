import numpy as np
from numpy import asarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from calculo_imagenes import *
from timeit import default_timer as timer

archivo = "G4.bmp"
replica = lectura(archivo)
U = asarray(replica)
width, height = replica.size

# N es el número de píxeles en el eje x de la imagen origen

N = height

# M es el número de pixeles en el eje y de la imagen de origen

M = width

archivo = "ref.bmp"
replica_ref = lectura(archivo)
ref = asarray(replica_ref)

parametro_d = 1.228e-3
W = 6.86e-3
L = 14e-3
mag = 1
lamb = 532e-9

U = U.astype(np.float32)
ref = ref.astype(np.float32)
U_gpu = gpuarray.to_gpu(U)
ref_gpu = gpuarray.to_gpu(ref)
mod = SourceModule("""
#include <math.h>
#include <stdio.h>
#include <cuComplex.h>

// siento que esta función es para ajustar los ejes, pero no estoy seguro
__global__ void Restando_Referencia(float *real, float *temp, int width, int height){

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	//Haciendo la resta...
	real[fila*width + col] = real[fila*width + col] - temp[fila*width + col];
	temp[fila*width + col] = 0;

}

//Qué es un remapeo para kreuser?

__global__ void Kreuzer_Remapeo(float *idata, float *idata_remap, int width, int height, float parametro_d, float L, float W)
{

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	/*Ac? aplicamos la trasformaci?n planteada por kreuzer para el holograma*/

	float Xo, Yo, deltaX, deltaY;
	float XoP, YoP, deltaXP, deltaYP;
	float L2;

	L2 = L*L;

	Xo = -W / 2.0;
	Yo = W / 2.0;

	deltaX = W / (width - 1);
	deltaY = -W / (height - 1);

	XoP = Xo * L / sqrt(L2 + Xo*Xo);
	YoP = Yo * L / sqrt(L2 + Xo*Xo);

	deltaXP = -2.0 * XoP / (width - 1);
	deltaYP = -2.0 * YoP / (height - 1);

	float deltaXm1 = 1.0 / deltaX;
	float deltaYm1 = 1.0 / deltaY;

	float YPos = YoP + fila*deltaYP;
	float XPos = XoP + col*deltaXP;
	float RPm1 = 1 / sqrt(L2 - XPos*XPos - YPos*YPos);

	float newXPos = XPos*L*RPm1;
	float newYPos = YPos*L*RPm1;

	float Xcoord = (newXPos - Xo)*deltaXm1;
	float Ycoord = (newYPos - Yo)*deltaYm1;

	int iXcoord = (int)floor(Xcoord);
	int iYcoord = (int)floor(Ycoord);

	float x1frac = (iXcoord + 1.0) - Xcoord;
	float x2frac = 1.0 - x1frac;
	float y1frac = (iYcoord + 1.0) - Ycoord;
	float y2frac = 1.0 - y1frac;

	float x1y1 = x1frac*y1frac;
	float x1y2 = x1frac*y2frac;
	float x2y1 = x2frac*y1frac;
	float x2y2 = x2frac*y2frac;

	/*
	//Inicializamos el array donde quedaran los datos del holograma remapeado
	idata_remap[ fila*width+col ].x = 0;
	*/

	/*
	idata[ fila*width+col ].x = idata[ fila*width+col ].x - promedio;

	*/

	//Teniendo todos los valores listos, ahora hacemos el "remapeo" sobre el holograma

	if (iYcoord>0 && iYcoord<height / 2 && iXcoord>0 && iXcoord<width / 2)
	{
		//Cuadrante 1
		idata_remap[fila*width + col] = ((x1y1*idata[iYcoord*width + iXcoord])
			+ (x2y1*idata[iYcoord*width + iXcoord + 1])
			+ (x1y2*idata[(iYcoord + 1)*width + iXcoord])
			+ (x2y2*idata[(iYcoord + 1)*width + iXcoord + 1]));

		//Cuadrante 2
		idata_remap[(fila + 1)*width - 1 - col] = ((x1y1*idata[(iYcoord + 1)*width - 1 - iXcoord])
			+ (x2y1*idata[(iYcoord + 1)*width - 1 - iXcoord - 1])
			+ (x1y2*idata[(iYcoord + 2)*width - 1 - iXcoord])
			+ (x2y2*idata[(iYcoord + 2)*width - 1 - iXcoord - 1]));

		//cudrante 3
		idata_remap[(height - 1 - fila)*width + col] = (x1y1*idata[(height - 1 - iYcoord)*width + iXcoord] +
			x2y1*idata[(height - 1 - iYcoord)*width + iXcoord + 1] +
			x1y2*idata[(height - 2 - iYcoord)*width + iXcoord] +
			x2y2*idata[(height - 2 - iYcoord)*width + iXcoord + 1]);

		//Cuadrante 4
		idata_remap[(height - fila)*width - col - 1] = (x1y1*idata[(height - iYcoord)*width - iXcoord - 1] +
			x2y1*idata[(height - iYcoord)*width - iXcoord - 2] +
			x1y2*idata[(height - 1 - iYcoord)*width - iXcoord - 1] +
			x2y2*idata[(height - 1 - iYcoord)*width - iXcoord - 2]);

	}
}



//que es esto?

__global__ void  generacion_f1_f2(float *idata_remap_real, float *idata_remap_imag, float *matriz_holo_real, float *matriz_holo_imag, int width, int height, float parametro_d, float M, float L, float W, float lambda)
{

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	// Parametros


	float deltax = W / width;

	float z = parametro_d;	

	float L2 = L*L;


	float k = 2 * (3.141592) / lambda;

	float deltaX = M*(z)*deltax / L;
	float deltaY = deltaX;

	//origen de coordenadas holograma
	float xo = -W / 2;

	//origen de coordenadas holograma transformado
	float xop = xo * L / sqrt(L2 + xo*xo);
	float yop = xop;

	float deltaxp = L / (width)* ((xo + (width - 1)*deltax) / sqrt(L2 + (xo + (width - 1)*deltax)*(xo + (width - 1)*deltax)) - xo / sqrt(L2 + xo*xo));
	float deltayp = deltaxp;

	//origen de coordenadas plano de reconstrucci?n
	float Yo = -deltaX*(width) / 2;
	float Xo = -deltaX*(width) / 2;

	//MULTIPLICAR HOLOGRAMA POR FASE DE PROPAGACI?N
	float Rp = sqrt((L2)-((deltaxp*col + xop)*(deltaxp*col + xop)) - ((deltayp*fila + yop)*(deltayp*fila + yop)));

	float argumento_1 = (L / Rp)*(L / Rp)*(L / Rp)*(L / Rp);
	float argumento_2 = k*z*Rp / L;	


	float Ip_real = (argumento_1)*__cosf(argumento_2);
	float Ip_imag = (argumento_1)*__sinf(argumento_2);


	float fase_real = __cosf((k / (2 * L))*(2 * Xo*col*deltaxp + 2 * Yo*fila*deltayp + (col*col)*deltaxp*deltaX + (fila*fila)*deltayp*deltaY));
	float fase_imag = __sinf((k / (2 * L))*(2 * Xo*col*deltaxp + 2 * Yo*fila*deltayp + (col*col)*deltaxp*deltaX + (fila*fila)*deltayp*deltaY));


	//funci?n f1
	//Cuadrante 1
	idata_remap_imag[fila*width + col] = ((idata_remap_real[fila*width + col] * Ip_real*fase_imag) + (idata_remap_real[fila*width + col]*Ip_imag*fase_real));
	idata_remap_real[fila*width + col] = ((idata_remap_real[fila*width + col] * Ip_real*fase_real) - (idata_remap_real[fila*width + col]*Ip_imag*fase_imag));

	/*
	//Cuadrante 2
	idata_remap[(fila+1)*width-1-col].y = ((idata_remap[(fila+1)*width-1-col].x*Ip_real*fase_imag)+(idata_remap[(fila+1)*width-1-col].x*Ip_imag*fase_real));
	idata_remap[(fila+1)*width-1-col].x = ((idata_remap[(fila+1)*width-1-col].x*Ip_real*fase_real)-(idata_remap[(fila+1)*width-1-col].x*Ip_imag*fase_imag));


	//Cuadrante 3
	idata_remap[(height-1-fila)*width+col].y = ((idata_remap[(height-1-fila)*width+col].x*Ip_real*fase_imag)+(idata_remap[(height-1-fila)*width+col].x*Ip_imag*fase_real));
	idata_remap[(height-1-fila)*width+col].x = ((idata_remap[(height-1-fila)*width+col].x*Ip_real*fase_real)-(idata_remap[(height-1-fila)*width+col].x*Ip_imag*fase_imag));


	//Cuadrante 4
	idata_remap[(height-fila)*width-col-1].y = ((idata_remap[(height-fila)*width-col-1].x*Ip_real*fase_imag)+(idata_remap[(height-fila)*width-col-1].x*Ip_imag*fase_real));
	idata_remap[(height-fila)*width-col-1].x = ((idata_remap[(height-fila)*width-col-1].x*Ip_real*fase_real)-(idata_remap[(height-fila)*width-col-1].x*Ip_imag*fase_imag));
	*/

	//funcion f2
	//Cuadrante 1
	matriz_holo_real[fila*width + col] = __cosf((k / (2 * L))*((col - width / 2)*(col - width / 2)*deltaxp*deltaX + (fila - width / 2)*(fila - width / 2)*deltayp*deltaY));
	matriz_holo_imag[fila*width + col] = (-1)*__sinf((k / (2 * L))*((col - width / 2)*(col - width / 2)*deltaxp*deltaX + (fila - width / 2)*(fila - width / 2)*deltayp*deltaY));

	/*
	//Cuadrante 2
	matriz_holo[(fila+1)*width-1-col].x = matriz_holo[fila*width+col].x;
	matriz_holo[(fila+1)*width-1-col].y = matriz_holo[fila*width+col].y;


	//Cuadrante 3
	matriz_holo[(height-1-fila)*width+col].x = matriz_holo[fila*width+col].x;
	matriz_holo[(height-1-fila)*width+col].y = matriz_holo[fila*width+col].y;


	//Cuadrante 4
	matriz_holo[(height-fila)*width-col-1].x = matriz_holo[fila*width+col].x;
	matriz_holo[(height-fila)*width-col-1].y = matriz_holo[fila*width+col].y;
	*/
}


__global__ void CambioTipoVariable(float *idata_remap_real, float *idata_remap_imag, float *matriz_holo_real, float *matriz_holo_imag, cuComplex *arreglo1, cuComplex *arreglo2, int width, int height)
{

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	arreglo1[(fila*width + col)].x = idata_remap_real[fila*width + col]; //En los pares meto los reales...
	arreglo1[((fila*width + col))].y = idata_remap_imag[fila*width + col]; //En los impares meto los imaginarios

	arreglo2[(fila*width + col)].x = matriz_holo_real[fila*width + col]; //En los pares meto los reales...
	arreglo2[((fila*width + col))].y = matriz_holo_imag[fila*width + col]; //En los impares meto los imaginarios

}

__global__ void CambioTipoVariable2(float *idata_remap_real, float *idata_remap_imag, float *matriz_holo_real, float *matriz_holo_imag, cuComplex *arreglo1, cuComplex *arreglo2, int width, int height)
{

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	idata_remap_real[fila*width + col] = arreglo1[(fila*width + col)].x; //En los pares meto los reales...
	idata_remap_imag[fila*width + col] = arreglo1[((fila*width + col))].y; //En los impares meto los imaginarios

	matriz_holo_real[fila*width + col] = arreglo2[(fila*width + col)].x; //En los pares meto los reales...
	matriz_holo_imag[fila*width + col] = arreglo2[((fila*width + col))].y; //En los impares meto los imaginarios

}

 __global__ void multiplicacion(float *idata_remap_real, float *idata_remap_imag, float *matriz_holo_real, float *matriz_holo_imag, float *d_temp13x, int width, int height)
{
	
	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	d_temp13x[fila*width + col] = ((idata_remap_real[fila*width + col])*(matriz_holo_real[fila*width + col])) -
		((idata_remap_imag[fila*width + col])*(matriz_holo_imag[fila*width + col]));

	matriz_holo_imag[fila*width + col] = ((idata_remap_imag[fila*width + col])*(matriz_holo_real[fila*width + col])) +
		((idata_remap_real[fila*width + col])*(matriz_holo_imag[fila*width + col]));

	matriz_holo_real[fila*width + col] = d_temp13x[fila*width + col];

}
__global__ void fft_shift(float *real, float *imag, float *d_temp13x, int width, int height)
{

	/*Variables que nos delimitan la operacion en CUDA (Para no coger todos los hilos disponibles,
	ya que esto implica m�s tiempo de operaci�n en CUDA*/

	int m2 = width / 2;
	int n2 = height / 2;

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;
	int col2 = blockIdx.x*blockDim.x + threadIdx.x + m2;
	int fila2 = blockIdx.y*blockDim.y + threadIdx.y + n2;


	/*Este condicional limita los hilos que se encargan de llevar a cabo el proceso, no es necesario por la simetr�a
	(matrix de N*N), pero en caso de no existir esta simetr�a es fundamental*/

	//if (col2 < m2 && col < m2 && fila2 < n2 && fila < m2) {   

	d_temp13x[fila*width + col] = real[fila*width + col];  //Guardo el primer cuadrante
	real[fila*width + col] = real[fila2*width + col2];  //en el primer cuadrante estoy poniendo lo que hay en el tercero
	real[fila2*width + col2] = d_temp13x[fila*width + col];//En el tercer cuadrante estoy poniendo lo que habia en el primero

	d_temp13x[fila*width + col] = imag[fila*width + col];  //Lo mismo anterior pero para los imaginarios
	imag[fila*width + col] = imag[fila2*width + col2];
	imag[fila2*width + col2] = d_temp13x[fila*width + col];

	d_temp13x[fila*width + col] = real[fila*width + col2];//Guardo Cuadrante dos
	real[fila*width + col2] = real[fila2*width + col];  //En el segundo guardo lo que hay en el cuarto
	real[fila2*width + col] = d_temp13x[fila*width + col];//En el cuarto guardo lo que estaba en el segundo

	d_temp13x[fila*width + col] = imag[fila*width + col2]; //Lo mismo que en el anterior
	imag[fila*width + col2] = imag[fila2*width + col];
	imag[fila2*width + col] = d_temp13x[fila*width + col];
	//}

}

__global__ void multiplicacion_fase(float *idata_real, float *idata_imag, float *odata_real, float *odata_imag, int width, int height, float parametro_d, float L, float W, float lambda)
{

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	// Par?metros


	float deltax = W / width;

	float z = parametro_d;

	float L2 = L*L;

	float k = 2 * (3.141592) / lambda;

	float deltaX = (z)*deltax / L;
	float deltaY = deltaX;

	//origen de coordenadas holograma
	float xo = -W / 2;

	//origen de coordenadas holograma transformado
	float xop = xo * L / sqrt(L2 + xo*xo);
	float yop = xop;

	float deltaxp = L / (width)* ((xo + (width - 1)*deltax) / sqrt(L2 + (xo + (width - 1)*deltax)*(xo + (width - 1)*deltax)) - xo / sqrt(L2 + xo*xo));
	float deltayp = deltaxp;

	//origen de coordenadas plano de reconstrucci?n
	float Yo = -deltaX*(width) / 2;
	float Xo = Yo;

	float termino_1 = (k / L)*((Xo + col*deltaX)*xop + (Yo + fila*deltaY)*yop);
	float termino_2 = (0.5*k / L)*((col - width / 2)*(col - width / 2)*deltaxp*deltaX + (fila - width / 2)*(fila - width / 2)*deltayp*deltaY);


	//Ac? empieza lo paralelo:
	float real = deltaxp*deltayp*((__cosf(termino_1) * __cosf(termino_2)) -
		(__sinf(termino_1) * __sinf(termino_2)));

	float imag = deltaxp*deltayp*((__cosf(termino_1) * __sinf(termino_2)) +
		(__sinf(termino_1) * __cosf(termino_2)));


	//Cuadrante 1
	odata_real[fila*width + col] = (idata_real[fila*width + col] * real) - (idata_imag[fila*width + col] * imag);
	odata_imag[fila*width + col] = (idata_imag[fila*width + col] * real) + (idata_real[fila*width + col] * imag);

	/*
	//Cuadrante 2
	idata_remap[(fila+1)*width-1-col].x = (idata[(fila+1)*width-1-col].x * real) - (idata[(fila+1)*width-1-col].y * imag);
	idata_remap[(fila+1)*width-1-col].y = (idata[(fila+1)*width-1-col].y * real) + (idata[(fila+1)*width-1-col].x * imag);


	//Cuadrante 3
	idata_remap[(height-1-fila)*width+col].x = (idata[(height-1-fila)*width+col].x * real) - (idata[(height-1-fila)*width+col].y * imag);
	idata_remap[(height-1-fila)*width+col].y = (idata[(height-1-fila)*width+col].y * real) + (idata[(height-1-fila)*width+col].x * imag);


	//Cuadrante 4
	idata_remap[(height-fila)*width-col-1].x = (idata[(height-fila)*width-col-1].x * real) - (idata[(height-fila)*width-col-1].y * imag);
	idata_remap[(height-fila)*width-col-1].y = (idata[(height-fila)*width-col-1].y * real) + (idata[(height-fila)*width-col-1].x * imag);
	*/


}

__global__ void getStats(float *pArray, float *pMaxResults, float *pMinResults)
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

__global__ void escalamiento(float *temp, int width, int height, float maximo, float minimo)
{

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	temp[fila*width + col] = (temp[fila*width + col]) - minimo;
	temp[fila*width + col] = (temp[fila*width + col]) / (maximo - minimo);
	temp[fila*width + col] = (temp[fila*width + col]) * 255;
	//Ac? tenemos todas los pixeles escalados a 8 bits (255 niveles de gris)

}

__global__ void amplitud(float *odata_real, float *odata_imag, float *temp_intensidad, int width, int height)
{
	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;

	//Calculo de la intensidad
	temp_intensidad[fila*width + col] = ((odata_real[fila*width + col])*(odata_real[fila*width + col])) +
		((odata_imag[fila*width + col])*(odata_imag[fila*width + col]));
		
	//odata_real[fila*width + col] = (temp_intensidad[fila*width + col]);
	odata_real[fila*width + col] = sqrt(temp_intensidad[fila*width + col]);

	//Ac? tenemos todas las intensidades resultantes

}
//pasamos a dos floats a un cuComplex para fft
__global__ void reubicacion(cuComplex *a, float *b, float *c,int N){

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;
	int i2= ((fila*N)+col);
	a[i2].x = b[i2];
	a[i2].y = c[i2];
}
//pasamos de un cuComplex a dos floats para fft
__global__ void desreubicacion(cuComplex *a, float *b, float *c,int N){

	//Descriptores de cada hilo
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int fila = blockIdx.y*blockDim.y + threadIdx.y;
	int i2= ((fila*N)+col);
	b[i2] = a[i2].x;
	c[i2] = a[i2].y;
}
__global__ void fft_shift_py(cuComplex *final,cuComplex *dest_gpu, int N, int M)
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

""")
Restando = mod.get_function("Restando_Referencia")
fase = mod.get_function("multiplicacion_fase")
remapeo = mod.get_function("Kreuzer_Remapeo")
f1_f2 = mod.get_function("generacion_f1_f2")
cambio1 = mod.get_function("CambioTipoVariable")
cambio2 = mod.get_function("CambioTipoVariable2")
multiplicacion = mod.get_function("multiplicacion")
fft_shift = mod.get_function("fft_shift")
fft_shift_py = mod.get_function("fft_shift_py")
amplitud = mod.get_function("amplitud")
stats = mod.get_function("getStats")
escalamiento = mod.get_function("escalamiento")
reubicacion = mod.get_function("reubicacion")
desreubicacion = mod.get_function("desreubicacion")

U_img = gpuarray.empty((N, M), np.float32)
ref_img = gpuarray.empty((N, M), np.float32)
var2_real = gpuarray.empty((N, M), np.float32)
var2_img = gpuarray.empty((N, M), np.float32)
maxi = gpuarray.empty((N, M), np.float32)
mini = gpuarray.empty((N, M), np.float32)
fft_sim = gpuarray.empty((N, M), np.complex64)
fft_sim_img = gpuarray.empty((N, M), np.complex64)
previo = gpuarray.empty((N, M), np.complex64)
fft = gpuarray.empty((N, M), np.complex64)

block_dim = (16, 16, 1)

# Mallado

grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)

Restando(U_gpu, ref_gpu, np.int32(M), np.int32(
    N), block=block_dim, grid=grid_dim)

mai = U_gpu.get()

# Se recrea la matriz de salida de el shift en la cpu
finale = mai.reshape((N, M))

mostrar(((np.abs(finale))), "Referencia menos original",
        "pixeles en el eje x", "pixeles en el eje y")

grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)

remapeo(U_gpu, ref_gpu, np.int32(N), np.int32(M),
        np.float32(parametro_d), np.float32(L), np.float32(W), block=block_dim, grid=grid_dim)
mai = ref_gpu.get()

# Se recrea la matriz de salida de el shift en la cpu
finale = mai.reshape((N, M))
mostrar(((np.abs(finale))), "Remapeo",
        "pixeles en el eje x", "pixeles en el eje y")

grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)

f1_f2(ref_gpu, ref_img, var2_real, var2_img, np.int32(N), np.int32(M),
      np.float32(parametro_d), np.float32(mag), np.float32(L), np.float32(W), np.float32(lamb), block=block_dim, grid=grid_dim)

mai = ref_img.get()

# Se recrea la matriz de salida de el shift en la cpu
finale = mai.reshape((N, M))
mostrar(((np.angle(finale))), "Función 1 Función 2",
        "pixeles en el eje x", "pixeles en el eje y")

cambio1(ref_gpu, ref_img, var2_real, var2_img, fft_sim, fft_sim_img,
        np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)
mai = ref_gpu.get()

# Se recrea la matriz de salida de el shift en la cpu
finale = mai.reshape((N, M))
mostrar(((np.angle(finale))), "Primer cambio",
        "pixeles en el eje x", "pixeles en el eje y")

plan = cu_fft.Plan(U.shape, np.complex64, np.complex64)

cu_fft.fft(fft_sim, fft, plan)
cu_fft.fft(fft_sim_img, previo, plan)

grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)
cambio2(U_gpu, ref_img, var2_real, var2_img, fft, previo,
        np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

multiplicacion(U_gpu, ref_img, var2_real, var2_img, ref_gpu, np.int32(N), np.int32(M),
               block=block_dim, grid=grid_dim)

cambio1(U_gpu, ref_img, var2_real, var2_img, fft, previo,
        np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

mai = fft.get()
finale = mai.reshape((N, M))
mostrar((np.log(np.abs(finale))), "pre-ifft",
        "pixeles en el eje x", "pixeles en el eje y")

cu_fft.ifft(previo, fft_sim_img, plan)

cambio2(U_gpu, ref_img, var2_real, var2_img, fft, fft_sim_img,
        np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)


grid_dim = (N // (block_dim[0]*2), M // (block_dim[1]*2), 1)

fft_shift(var2_real, var2_img, ref_gpu, np.int32(N),
          np.int32(M), block=block_dim, grid=grid_dim)

grid_dim = (N // (block_dim[0]), M // (block_dim[1]), 1)

fase(var2_real, var2_img, U_gpu, ref_img, np.int32(N), np.int32(M), np.float32(
    parametro_d), np.float32(L), np.float32(W), np.float32(lamb), block=block_dim, grid=grid_dim)

cambio1(U_gpu, ref_img, var2_real, var2_img, fft, previo,
        np.int32(N), np.int32(M), block=block_dim, grid=grid_dim)

mai = fft.get()
finale = mai.reshape((N, M))
mostrar(((np.angle(finale))), "pre-final",
        "pixeles en el eje x", "pixeles en el eje y")

amplitud(U_gpu, ref_img, ref_gpu, np.int32(N),
         np.int32(M), block=block_dim, grid=grid_dim)

stats(U_gpu, maxi,
      mini, block=block_dim, grid=grid_dim)


mai = maxi.get()
finale = mai.reshape((N, M))
mai2 = mini.get()
finale2 = mai.reshape((N, M))
minimo = np.min(finale)
maximo = np.max(finale2)

escalamiento(U_gpu, np.int32(N), np.int32(M), np.float32(maximo),
             np.float32(minimo), block=block_dim, grid=grid_dim)
print(minimo)
print(maximo)
mai = U_gpu.get()
finale = mai.reshape((N, M))
mostrar((np.power(np.abs(finale), 2)), "final",
        "pixeles en el eje x", "pixeles en el eje y")
