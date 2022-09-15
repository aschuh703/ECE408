#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

const int MASK_WIDTH = 3;
const int MASK_RADIUS = MASK_WIDTH / 2;
const int TILE_SIZE = 4;
const int BLOCK_SIZE = TILE_SIZE + MASK_WIDTH - 1;

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *A, float *B, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here

  __shared__ float A_s[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

  const int out_z = blockIdx.z * TILE_SIZE + threadIdx.z;
  const int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
  const int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;

  const int in_z = out_z - MASK_RADIUS;
  const int in_y = out_y - MASK_RADIUS;
  const int in_x = out_x - MASK_RADIUS;

  float output = 0.0;

  if (in_z >= 0 && in_z < z_size && in_y >= 0 && in_y < y_size &&
      in_x >= 0 && in_x < x_size) {
    A_s[threadIdx.z][threadIdx.y][threadIdx.x] =
        A[in_z * (y_size * x_size) + in_y * x_size + in_x];
  } else {
    A_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();

  if (threadIdx.z < TILE_SIZE && threadIdx.y < TILE_SIZE &&
      threadIdx.x < TILE_SIZE) {
    for (int z = 0; z < MASK_WIDTH; ++z) {
      for (int y = 0; y < MASK_WIDTH; ++y) {
        for (int x = 0; x < MASK_WIDTH; ++x) {
          output += deviceKernel[z][y][x] *
                    A_s[z + threadIdx.z][y + threadIdx.y][x + threadIdx.x];
        }
      }
    }

    if (out_z < z_size && out_y < y_size && out_x < x_size) {
      B[out_z * (y_size * x_size) + out_y * x_size + out_x] = output;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // Not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput,
                     &hostInput[3], // first three elements are dimensions
                     (inputLength - 3) * sizeof(float),
                     cudaMemcpyHostToDevice););
  wbCheck(cudaMemcpyToSymbol(deviceKernel, hostKernel,
                             MASK_WIDTH * MASK_WIDTH * MASK_WIDTH *
                                 sizeof(float)););
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(ceil(((float)x_size) / TILE_SIZE),
               ceil(((float)y_size) / TILE_SIZE),
               ceil(((float)z_size) / TILE_SIZE));

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size,
                                x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput,
             (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
