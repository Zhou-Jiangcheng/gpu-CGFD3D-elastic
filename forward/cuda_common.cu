#include <stdio.h>
#include "cuda_common.h"

void setDeviceBeforeInit()
{
  char *localRankStr = NULL;
  int rank = 0, devCount = 0;
  if (NULL != (localRankStr = getenv(ENV_LOCAL_RANK))){
    rank = atoi(localRankStr);
  }
  CUDACHECK(cudaGetDeviceCount(&devCount));
  CUDACHECK(cudaSetDevice((rank%devCount)));
  //debuge
  fprintf(stdout,"rank is %d\n",rank);
  fprintf(stdout,"device count is %d\n",devCount);
  fflush(stdout);
}

void *cuda_malloc(size_t len)
{
  void *p;
  const cudaError_t err = cudaMalloc(&p, len);
  if (cudaSuccess == err) return p;
  fprintf(stderr, "Error @ %s, ", __FILE__);
  fprintf(stderr, "code: %d, reson: %s\n", err, cudaGetErrorString(err));
  return 0;
}



