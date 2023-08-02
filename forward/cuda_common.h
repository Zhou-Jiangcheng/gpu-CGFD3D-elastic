#ifndef COMMON_H
#define COMMON_H

//#define ENV_LOCAL_RANK "MV2_COMM_WORLD_LOCAL_RANK"
#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"

#define CUDACHECK(call) {                                    \
  const cudaError_t error = call;                            \
  if (error != cudaSuccess) {                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n",              \
          error, cudaGetErrorString(error));                 \
  }                                                          \
}

void setDeviceBeforeInit(int gpu_id_start);
  
void *cuda_malloc(size_t len);

#endif

