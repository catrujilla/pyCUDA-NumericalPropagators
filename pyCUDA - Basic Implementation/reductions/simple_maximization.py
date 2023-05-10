import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

from timeit import default_timer as timer

#import imageio
import imageio.v2 as imageio

# CUDA kernel for reduction
mod = SourceModule("""
__global__ void max_reduce(int *input, int *output, int width, int height)
{
    __shared__ int sdata[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    sdata[tid] = -1;

    //Find the max value of local array
    for (int j = 0; j < height; j++) {
        if (i < width) {
            sdata[tid] = max(sdata[tid], input[i + j * width]);
        }
    }
    __syncthreads();

    // Reduce the shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] = max(sdata[index], sdata[index + s]);
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
""")



# Input data
#input_data = np.random.randint(0, 100, (4096, 4096)) #Random data
# Read in the image file
input_data = imageio.imread('holonotele4cm8x.tif')

input_data = input_data.astype(np.uint32)

print (input_data.nbytes)
print (input_data.dtype)

#CPU version
start = timer()	#Start to count time

# Find the maximum value
max_val = np.amax(input_data)
#max_val = max(map(max, input_data))

print("Procesing time, CPU: ", timer()-start) #Time for get_plus1 execution
print("Maximum value:", max_val)

#GPU version

# Copy input data to the GPU
input_gpu = drv.mem_alloc(input_data.nbytes)
drv.memcpy_htod(input_gpu, input_data)

# Maximum function
max_reduce = mod.get_function("max_reduce")
width, height = input_data.shape
output_data = np.zeros((height)).astype(np.uint32)
#print (output_data.shape)

output_gpu = drv.mem_alloc(output_data.nbytes)
#print (output_data.nbytes)

# Number of blocks and threads
block = (16, 16, 1)
grid = (width // block[0], height // block[1])

start = timer()	#Start to count time

# kernel function
max_reduce(input_gpu, output_gpu, np.int32(width), np.int32(height), block=block, grid=grid)

print("Procesing time, GPU: ", timer()-start) #Time for get_plus1 execution
print("Maximum value:", max_val)

start = timer()	#Start to count time

# Copy the results back to the host
drv.memcpy_dtoh(output_data, output_gpu)

print("Procesing time, GPU: ", timer()-start) #Time for get_plus1 execution

start = timer()	#Start to count time

# Get the maximum value
max_val = output_data.max()
#max_val = np.amax(output_data)

print("Procesing time, GPU: ", timer()-start) #Time for get_plus1 execution



