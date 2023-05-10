/*
 * Proyecto en el cual se lleva a cabo la reconstrucción de un holograma de entrada (imagen en el disco)
 * via JCUDA y se visualiza en un archivo de salida en el disco
 */
package jcudareconstruccionholograma;

import java.awt.image.BufferedImage;
import static jcuda.driver.JCudaDriver.*;

import java.io.*;
import javax.imageio.ImageIO;

import jcuda.*;
import jcuda.driver.*;
import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;

/**
 *
 * @author Opto-Digital Carlos Alejandro Trujillo Anaya
 */
public class JCUDAReconstruccionHolograma {

        /**
         * @param args the command line arguments
         */
        public static void main(String[] args) throws IOException {

                // Enable exceptions and omit all subsequent error checks
                JCudaDriver.setExceptionsEnabled(true);

                // Create the PTX file by calling the NVCC
                String ptxFileName = preparePtxFile("kernels.cu");

                // Initialize the driver and create a context for the first device.
                cuInit(0);
                CUdevice device = new CUdevice();
                cuDeviceGet(device, 0);
                CUcontext context = new CUcontext();
                cuCtxCreate(context, 0, device);

                // Load the ptx file.
                CUmodule module = new CUmodule();
                cuModuleLoad(module, ptxFileName);

                // Declaración de los kernels para ser llamados desde JCUDA:
                // Obtain a function pointer to the "Restando_Referencia" function.
                CUfunction functionRestando_Referencia = new CUfunction();
                cuModuleGetFunction(functionRestando_Referencia, module, "Restando_Referencia");
                // Obtain a function pointer to the "getStats" function.
                CUfunction functiongetStats = new CUfunction();
                cuModuleGetFunction(functiongetStats, module, "getStats");
                // Obtain a function pointer to the "escalamiento" function.
                CUfunction functionescalamiento = new CUfunction();
                cuModuleGetFunction(functionescalamiento, module, "escalamiento");
                // Obtain a function pointer to the "Kreuzer_Remapeo" function.
                CUfunction functionKreuzer_Remapeo = new CUfunction();
                cuModuleGetFunction(functionKreuzer_Remapeo, module, "Kreuzer_Remapeo");
                // Obtain a function pointer to the "generacion_f1_f2" function.
                CUfunction functiongeneracion_f1_f2 = new CUfunction();
                cuModuleGetFunction(functiongeneracion_f1_f2, module, "generacion_f1_f2");
                // Obtain a function pointer to the "CambioTipoVariable" function.
                CUfunction functionCambioTipoVariable = new CUfunction();
                cuModuleGetFunction(functionCambioTipoVariable, module, "CambioTipoVariable");
                // Obtain a function pointer to the "CambioTipoVariable2" function.
                CUfunction functionCambioTipoVariable2 = new CUfunction();
                cuModuleGetFunction(functionCambioTipoVariable2, module, "CambioTipoVariable2");
                // Obtain a function pointer to the "multiplicacion" function.
                CUfunction functionmultiplicacion = new CUfunction();
                cuModuleGetFunction(functionmultiplicacion, module, "multiplicacion");
                // Obtain a function pointer to the "fft_shift" function.
                CUfunction functionfft_shift = new CUfunction();
                cuModuleGetFunction(functionfft_shift, module, "fft_shift");
                // Obtain a function pointer to the "multiplicacion_fase" function.
                CUfunction functionmultiplicacion_fase = new CUfunction();
                cuModuleGetFunction(functionmultiplicacion_fase, module, "multiplicacion_fase");
                // Obtain a function pointer to the "modulo" function.
                CUfunction functionmodulo = new CUfunction();
                cuModuleGetFunction(functionmodulo, module, "modulo");
                // Obtain a function pointer to the "amplitud" function.
                CUfunction functionamplitud = new CUfunction();
                cuModuleGetFunction(functionamplitud, module, "amplitud");

                // Cargar la imagen sobre la cual vamos a trabajar
                // BufferedImage holograma = ImageIO.read(new File("HoloMHDL.bmp"));
                // BufferedImage referencia = ImageIO.read(new File("refMHDL.bmp"));
                // BufferedImage holograma = ImageIO.read(new File("G4.bmp"));
                // BufferedImage referencia = ImageIO.read(new File("GREF4.bmp"));

                // obtiene la imagen de entrada y la que debería obtener supongo
                BufferedImage holograma = ImageIO.read(new File("HoloMHDL1024.bmp"));
                BufferedImage referencia = ImageIO.read(new File("refMHDL1024.bmp"));

                // //Definición de parámetros (paramecios):
                // float parametro_d = (float) 1.68e-3;
                // float W = (float) 9.2e-3;
                // float L = (float) 20e-3;
                // float M = 1; //Magnificación
                // float lambda = (float) 532e-9;

                float parametro_d = (float) 4.5e-3;
                float W = (float) 6.86e-3;
                float L = (float) 19e-3;
                float M = 1; // Magnificación
                float lambda = (float) 405e-9;

                /*
                 * //Definición de parámetros (Mosca 2):
                 * float parametro_d = (float) 1.136e-3;
                 * //float W = (float) 8.57e-3;
                 * float W = (float) 6.86e-3;
                 * float L = (float) 14e-3;
                 * float M = 1; //Magnificación
                 * float lambda = (float) 532e-9;
                 * 
                 */
                // numero de pixeles
                int numElements = holograma.getWidth() * holograma.getHeight();
                // Variables para la reduccion (busqueda del maximo y minimo)
                int THREADS_PER_BLOCK = 256;
                int BLOCKS_PER_GRID_ROW = 128;

                // Allocate and fill the host input data
                float real[] = new float[numElements];
                float imag[] = new float[numElements];
                float real2[] = new float[numElements];
                float imag2[] = new float[numElements];
                float temp[] = new float[numElements];
                float h_resultMax[] = new float[numElements / THREADS_PER_BLOCK * Sizeof.FLOAT];
                float h_resultMin[] = new float[numElements / THREADS_PER_BLOCK * Sizeof.FLOAT];

                // Cargamos la info de la imagen en arrays para inciar el procesamiento
                for (int x = 0; x < holograma.getWidth(); x++) {
                        for (int y = 0; y < holograma.getHeight(); y++) {
                                int gris = (holograma.getRGB(x, y) >> 8) & 255;
                                int gris_ref = (referencia.getRGB(x, y) >> 8) & 255;
                                // real[y * N + x] = (float) (gris - gris_ref);
                                // real[y*N+x]=image.getRGB(x,y);
                                real[y * holograma.getHeight() + x] = (float) (gris);
                                imag[y * holograma.getHeight() + x] = 0;
                                real2[y * holograma.getHeight() + x] = (float) 0;
                                imag2[y * holograma.getHeight() + x] = (float) 0;
                                temp[y * holograma.getHeight() + x] = (float) (gris_ref);
                        }
                }

                // Allocate the device input data, and copy the
                // host input data to the device
                CUdeviceptr devicereal = new CUdeviceptr();
                cuMemAlloc(devicereal, numElements * Sizeof.FLOAT);
                cuMemcpyHtoD(devicereal, Pointer.to(real), numElements * Sizeof.FLOAT);
                CUdeviceptr deviceimag = new CUdeviceptr();
                cuMemAlloc(deviceimag, numElements * Sizeof.FLOAT);
                cuMemcpyHtoD(deviceimag, Pointer.to(imag), numElements * Sizeof.FLOAT);

                // Allocate device temp memory
                CUdeviceptr devicetemp = new CUdeviceptr();
                cuMemAlloc(devicetemp, numElements * Sizeof.FLOAT);
                cuMemcpyHtoD(devicetemp, Pointer.to(temp), numElements * Sizeof.FLOAT);

                // Generamos otra pareja de reales e imaginarios
                CUdeviceptr devicereal2 = new CUdeviceptr();
                cuMemAlloc(devicereal2, numElements * Sizeof.FLOAT);
                cuMemcpyHtoD(devicereal2, Pointer.to(real2), numElements * Sizeof.FLOAT);
                CUdeviceptr deviceimag2 = new CUdeviceptr();
                cuMemAlloc(deviceimag2, numElements * Sizeof.FLOAT);
                cuMemcpyHtoD(deviceimag2, Pointer.to(imag2), numElements * Sizeof.FLOAT);

                // Allocate device memory para este proceso
                CUdeviceptr d_resultMax = new CUdeviceptr();
                cuMemAlloc(d_resultMax, numElements / THREADS_PER_BLOCK * Sizeof.FLOAT);
                CUdeviceptr d_resultMin = new CUdeviceptr();
                cuMemAlloc(d_resultMin, numElements / THREADS_PER_BLOCK * Sizeof.FLOAT);

                // Organizamos los datos para las transformadas
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                // Generamos los buffers que se meten en la JCUFFT (la cual no acepta
                // cufftComplex como en CUFFT)
                CUdeviceptr bufferFFT1 = new CUdeviceptr();
                cuMemAlloc(bufferFFT1, numElements * Sizeof.FLOAT * 2); // Variable compleja para la JCUFFT
                CUdeviceptr bufferFFT2 = new CUdeviceptr();
                cuMemAlloc(bufferFFT2, numElements * Sizeof.FLOAT * 2);// Variable compleja para la JCUFFT

                // Solo trabajamos con x, porque en y tendremos el mismo tamaño: nuetras
                // imagenes son cuadradas (si no lo son, esto hay que cambiarlo!)
                int blockSizeX = 16; // Máximo tamaño que funciona bien!
                int gridSizeX_medio = (int) Math.ceil(holograma.getWidth() / (2 * blockSizeX)); // Ese dos es porque
                                                                                                // trabajamos sobre un
                                                                                                // cuarto de la imagen y
                                                                                                // el resto es copia y
                                                                                                // pegue
                int gridSizeX = (int) Math.ceil(holograma.getWidth() / (blockSizeX));
                /*
                 * #define THREADS_PER_BLOCK 256
                 * #define BLOCKS_PER_GRID_ROW 128
                 */
                int blockGridWidth = BLOCKS_PER_GRID_ROW;
                int blockGridHeight = (numElements / THREADS_PER_BLOCK) / blockGridWidth;

                long TInicio, TFin, tiempo; // Variables para determinar el tiempo de ejecución
                TInicio = System.currentTimeMillis(); // Tomamos la hora en que inicio el algoritmo y la almacenamos en
                                                      // la variable inicio

                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                Pointer kernelParametersRestarReferencia = Pointer.to(
                                Pointer.to(devicereal),
                                Pointer.to(devicetemp),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }));

                // Call the kernel function.
                cuLaunchKernel(functionRestando_Referencia,
                                gridSizeX, gridSizeX, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotros trabajamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersRestarReferencia, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // Ahora el Remapeo
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                Pointer kernelParametersKreuzer_Remapeo = Pointer.to(
                                Pointer.to(devicereal),
                                Pointer.to(devicetemp),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }),
                                Pointer.to(new float[] { parametro_d }),
                                Pointer.to(new float[] { L }),
                                Pointer.to(new float[] { W }));

                cuLaunchKernel(functionKreuzer_Remapeo,
                                gridSizeX_medio, gridSizeX_medio, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersKreuzer_Remapeo, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // Generación de f1 y f2
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                Pointer kernelParametersgeneracion_f1_f2 = Pointer.to(
                                Pointer.to(devicetemp), // Porque la funcion anterior deja la informacion en el temporal
                                Pointer.to(deviceimag),
                                Pointer.to(devicereal2),
                                Pointer.to(deviceimag2),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }),
                                Pointer.to(new float[] { parametro_d }),
                                Pointer.to(new float[] { M }),
                                Pointer.to(new float[] { L }),
                                Pointer.to(new float[] { W }),
                                Pointer.to(new float[] { lambda }));

                // Call the kernel function.
                cuLaunchKernel(functiongeneracion_f1_f2,
                                gridSizeX, gridSizeX, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersgeneracion_f1_f2, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                Pointer kernelParametersfunctionCambioTipoVariable = Pointer.to(
                                Pointer.to(devicetemp), // Porque la funcion anterior deja la informacion en el temporal
                                Pointer.to(deviceimag),
                                Pointer.to(devicereal2),
                                Pointer.to(deviceimag2),
                                Pointer.to(bufferFFT1),
                                Pointer.to(bufferFFT2),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }));

                // Call the kernel function.
                cuLaunchKernel(functionCambioTipoVariable,
                                gridSizeX, gridSizeX, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersfunctionCambioTipoVariable, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // ahora ejecutaremos las FFT de f1 y f2 con JCUFFT
                // Primero, hacemos el plan
                cufftHandle plan = new cufftHandle();
                JCufft.cufftPlan2d(plan, holograma.getWidth(), holograma.getHeight(), cufftType.CUFFT_C2C);

                // FFT de f1
                JCufft.cufftExecC2C(plan, bufferFFT1, bufferFFT1, JCufft.CUFFT_FORWARD);
                // FFt de f2
                JCufft.cufftExecC2C(plan, bufferFFT2, bufferFFT2, JCufft.CUFFT_FORWARD);

                // Multiplicamos punto a punto las matrices
                // Primero, las volvemos del tipo real por un lado e imag por el otro
                Pointer kernelParametersfunctionCambioTipoVariable2 = Pointer.to(
                                Pointer.to(devicereal),
                                Pointer.to(deviceimag),
                                Pointer.to(devicereal2),
                                Pointer.to(deviceimag2),
                                Pointer.to(bufferFFT1),
                                Pointer.to(bufferFFT2),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }));

                // Call the kernel function.
                cuLaunchKernel(functionCambioTipoVariable2,
                                gridSizeX, gridSizeX, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersfunctionCambioTipoVariable2, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // y ahora si, multiplicamos punto a punto
                Pointer kernelParametersfunctionmultiplicacion = Pointer.to(
                                Pointer.to(devicereal),
                                Pointer.to(deviceimag),
                                Pointer.to(devicereal2),
                                Pointer.to(deviceimag2),
                                Pointer.to(devicetemp),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }));

                // Call the kernel function.
                cuLaunchKernel(functionmultiplicacion,
                                gridSizeX, gridSizeX, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersfunctionmultiplicacion, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // Ejecutamos la IFFT, primero convertimos al tipo de JCUFFT
                // Call the kernel function.
                Pointer kernelParametersfunctionCambioTipoVariableOtra = Pointer.to(
                                Pointer.to(devicereal),
                                Pointer.to(deviceimag),
                                Pointer.to(devicereal2),
                                Pointer.to(deviceimag2),
                                Pointer.to(bufferFFT1),
                                Pointer.to(bufferFFT2),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }));

                cuLaunchKernel(functionCambioTipoVariable,
                                gridSizeX, gridSizeX, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersfunctionCambioTipoVariableOtra, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // y ahora si la IFFT
                // Como todo quedo en el segundo arreglo, trabajmos sobre él
                JCufft.cufftExecC2C(plan, bufferFFT2, bufferFFT2, JCufft.CUFFT_INVERSE);

                // Volvemos a convertir a nuestro tipo: real e imag separados
                Pointer kernelParametersfunctionCambioTipoVariable2Otra = Pointer.to(
                                Pointer.to(devicereal),
                                Pointer.to(deviceimag),
                                Pointer.to(devicereal2),
                                Pointer.to(deviceimag2),
                                Pointer.to(bufferFFT1),
                                Pointer.to(bufferFFT2),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }));
                // Call the kernel function.
                cuLaunchKernel(functionCambioTipoVariable2,
                                gridSizeX, gridSizeX, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersfunctionCambioTipoVariable2Otra, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // Aplicamos un fft shift
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                // Lo ejecutamos sobre el segundo buffer porque todo quedó en él
                Pointer kernelParametersfft_shift = Pointer.to(
                                Pointer.to(devicereal2),
                                Pointer.to(deviceimag2),
                                Pointer.to(devicetemp),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }));

                cuLaunchKernel(functionfft_shift,
                                gridSizeX_medio, gridSizeX_medio, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersfft_shift, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // Multiplicamos por términos de fase adicionales
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                Pointer kernelParametersmultiplicacion_fase = Pointer.to(
                                Pointer.to(devicereal2),
                                Pointer.to(deviceimag2),
                                Pointer.to(devicereal),
                                Pointer.to(deviceimag),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }),
                                Pointer.to(new float[] { parametro_d }),
                                Pointer.to(new float[] { L }),
                                Pointer.to(new float[] { W }),
                                Pointer.to(new float[] { lambda }));

                cuLaunchKernel(functionmultiplicacion_fase,
                                gridSizeX, gridSizeX, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersmultiplicacion_fase, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // Obtenemos el módulo (o la amplitud)
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                Pointer kernelParametersamplitud = Pointer.to(
                                Pointer.to(devicereal),
                                Pointer.to(deviceimag),
                                Pointer.to(devicetemp),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }));

                cuLaunchKernel(functionamplitud,
                                gridSizeX, gridSizeX, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersamplitud, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // Vamos a iniciar el proceso para llevar a cabo la busqueda del maximo y el
                // minimo sobre un arreglo de float
                // The max data size must be an integer multiple of 256*128, because each block
                // will have 256 threads,
                // and the block grid width will be 256. These are arbitrary numbers I choose.<=
                // El tipo escogió otros
                // valores, yo escogí estos porque me sirven a mi
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                Pointer kernelParametersgetStats = Pointer.to(
                                Pointer.to(devicereal),
                                Pointer.to(d_resultMax),
                                Pointer.to(d_resultMin));

                // Este algoritmo sirve para hallar el máximo y el mìnimo de un gran array como
                // es nuestro caso (4mb), en el sólo se copio la parte para le min
                // y el max, aunque tambien se podia sacar la media. Falta optimizar
                cuLaunchKernel(functiongetStats,
                                blockGridWidth, blockGridHeight, 1, // Grid dimension
                                THREADS_PER_BLOCK, 1, 1,
                                0, null, // Shared memory size and stream (Esto es para la memoria compartida dinamica,
                                         // la que utilizaremos sera la estatica por eso la definimos en el kernel mismo
                                         // )
                                kernelParametersgetStats, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // Copy the data back to the host
                cuMemcpyDtoH(Pointer.to(h_resultMin), d_resultMin,
                                numElements / THREADS_PER_BLOCK * Sizeof.FLOAT);
                cuMemcpyDtoH(Pointer.to(h_resultMax), d_resultMax,
                                numElements / THREADS_PER_BLOCK * Sizeof.FLOAT);

                // Each block returned one result, so lets finish this off with the cpu.
                // By using CUDA, we basically reduced how much the CPU would have to work by
                // about 256 times.
                float minimo = h_resultMin[0];
                float maximo = h_resultMax[0];

                for (int i = 1; i < numElements / THREADS_PER_BLOCK; i++) {
                        if (h_resultMin[i] < minimo) {
                                minimo = h_resultMin[i];
                        }
                        if (h_resultMax[i] > maximo) {
                                maximo = h_resultMax[i];
                        }

                }

                // Ahora con el minimo y el maximo, llevemos a cabo el escalamiento
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                Pointer kernelParametersescalamiento = Pointer.to(
                                Pointer.to(devicereal),
                                Pointer.to(new int[] { holograma.getWidth() }),
                                Pointer.to(new int[] { holograma.getHeight() }),
                                Pointer.to(new float[] { maximo }),
                                Pointer.to(new float[] { minimo }));

                cuLaunchKernel(functionescalamiento,
                                gridSizeX, gridSizeX, 1, // Grid dimension
                                blockSizeX, blockSizeX, 1, // Block dimension (x,y,z) (nosotro strabjamos con imagenes,
                                                           // por eso solo x e y)
                                0, null, // Shared memory size and stream
                                kernelParametersescalamiento, null // Kernel- and extra parameters
                );
                cuCtxSynchronize();

                // Allocate host output memory and copy the device output
                // to the host.
                cuMemcpyDtoH(Pointer.to(real), devicereal,
                                numElements * Sizeof.FLOAT);

                TFin = System.currentTimeMillis(); // Tomamos la hora en que finalizó el algoritmo y la almacenamos en
                                                   // la variable T
                tiempo = TFin - TInicio; // Calculamos los milisegundos de diferencia
                // System.out.println("Tiempo de ejecución en milisegundos: " + tiempo);
                // //Mostramos en pantalla el tiempo de ejecución en milisegundos
                System.out.println("Tiempo de ejecución en milisegundos: " + tiempo); // Mostramos en pantalla el tiempo
                                                                                      // de ejecución en milisegundos

                // Vamos a pasar todo esto a la imagen de salida para ir viendo que vamos
                // haciendo
                for (int x = 0; x < holograma.getWidth(); x++) {
                        for (int y = 0; y < holograma.getHeight(); y++) {
                                int pixel;

                                // real[y * holograma.getHeight() + x] = (float) (gris);
                                // imag[y * holograma.getHeight() + x] = 0;
                                pixel = (int) real[y * holograma.getHeight() + x];

                                // Tratamiento para la imagen de salida:
                                // En este caso la imagen de salida será de la misma natiraleza de la imagen
                                // de entrada, es decir, imagen de 8bits a escala de grises.
                                pixel = (pixel << 24) + (pixel << 16) + (pixel << 8) + pixel;
                                holograma.setRGB(x, y, pixel);
                        }
                }

                // Función para configurar los parámetros de la imagen de salida y crearla
                ImageIO.write(holograma, "JPG", new File("reconstruccion.jpg"));

                // Clean up.
                JCufft.cufftDestroy(plan);
                cuMemFree(devicereal);
                cuMemFree(deviceimag);
                cuMemFree(devicetemp);
                cuMemFree(devicereal2);
                cuMemFree(deviceimag2);
                cuMemFree(d_resultMax);
                cuMemFree(d_resultMin);
                cuMemFree(bufferFFT1);
                cuMemFree(bufferFFT2);

        }

        /**
         * The extension of the given file name is replaced with "ptx". If the file
         * with the resulting name does not exist, it is compiled from the given
         * file using NVCC. The name of the PTX file is returned.
         *
         * @param cuFileName The name of the .CU file
         * @return The name of the PTX file
         * @throws IOException If an I/O error occurs
         */
        private static String preparePtxFile(String cuFileName) throws IOException {
                int endIndex = cuFileName.lastIndexOf('.');
                if (endIndex == -1) {
                        endIndex = cuFileName.length() - 1;
                }
                String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
                File ptxFile = new File(ptxFileName);
                if (ptxFile.exists()) {
                        return ptxFileName;
                }

                File cuFile = new File(cuFileName);
                if (!cuFile.exists()) {
                        throw new IOException("Input file not found: " + cuFileName);
                }
                String modelString = "-m" + System.getProperty("sun.arch.data.model");
                String command = "nvcc " + modelString + " -ptx "
                                + cuFile.getPath() + " -o " + ptxFileName;

                System.out.println("Executing\n" + command);
                Process process = Runtime.getRuntime().exec(command);

                String errorMessage = new String(toByteArray(process.getErrorStream()));
                String outputMessage = new String(toByteArray(process.getInputStream()));
                int exitValue = 0;
                try {
                        exitValue = process.waitFor();
                } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new IOException(
                                        "Interrupted while waiting for nvcc output", e);
                }

                if (exitValue != 0) {
                        System.out.println("nvcc process exitValue " + exitValue);
                        System.out.println("errorMessage:\n" + errorMessage);
                        System.out.println("outputMessage:\n" + outputMessage);
                        throw new IOException(
                                        "Could not create .ptx file: " + errorMessage);
                }

                System.out.println("Finished creating PTX file");
                return ptxFileName;
        }

        /**
         * Fully reads the given InputStream and returns it as a byte array
         *
         * @param inputStream The input stream to read
         * @return The byte array containing the data from the input stream
         * @throws IOException If an I/O error occurs
         */
        private static byte[] toByteArray(InputStream inputStream)
                        throws IOException {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                byte buffer[] = new byte[8192];
                while (true) {
                        int read = inputStream.read(buffer);
                        if (read == -1) {
                                break;
                        }
                        baos.write(buffer, 0, read);
                }
                return baos.toByteArray();
        }

}
