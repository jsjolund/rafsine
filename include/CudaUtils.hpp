#pragma once

#include <cuda.h>
#include <math.h>
#include <math_constants.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "Eigen/Geometry"

#include "CudaMathHelper.h"

// Macro overloading
#define OVERLOADED_MACRO(M, ...) _OVR(M, _COUNT_ARGS(__VA_ARGS__))(__VA_ARGS__)
#define _OVR(macroName, number_of_args) _OVR_EXPAND(macroName, number_of_args)
#define _OVR_EXPAND(macroName, number_of_args) macroName##number_of_args
#define _COUNT_ARGS(...) \
  _ARG_PATTERN_MATCH(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define _ARG_PATTERN_MATCH(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N

// Define how to index memory
#define I3D(...) OVERLOADED_MACRO(I3D, __VA_ARGS__)
#define I3D6(x, y, z, nx, ny, nz) ((x) + (y) * (nx) + (z) * (nx) * (ny))
#define I3D2(pos, size) \
  (pos.x() + pos.y() * size.x() + pos.z() * size.x() * size.y())

#define I4D(...) OVERLOADED_MACRO(I4D, __VA_ARGS__)
#define I4D7(i, x, y, z, nx, ny, nz) \
  ((i) * (nx) * (ny) * (nz) + (x) + (y) * (nx) + (z) * (nx) * (ny))
#define I4D3(i, pos, size)                                             \
  ((i)*size.x() * size.y() * size.z() + pos.x() + pos.y() * size.x() + \
   pos.z() * size.x() * size.y())

#define df3D(i, x, y, z, nx, ny, nz) (df[I4D(i, x, y, z, nx, ny, nz)])
#define dftmp3D(i, x, y, z, nx, ny, nz) (df_tmp[I4D(i, x, y, z, nx, ny, nz)])

#define Tdf3D(i, x, y, z, nx, ny, nz) (dfT[I4D(i, x, y, z, nx, ny, nz)])
#define Tdftmp3D(i, x, y, z, nx, ny, nz) (dfT_tmp[I4D(i, x, y, z, nx, ny, nz)])

// Define the precision used for describing real number
typedef float real;
typedef float3 real3;
#define make_real3 make_float3
#define REAL_NAN CUDART_NAN_F
#define REAL_MAX FLT_MAX
#define REAL_MIN FLT_MIN

// Define a function as callable by both host CPU and CUDA device
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

/// use this if you want a 1D index
inline __device__ int idx1d(void) {
  return blockIdx.y * (gridDim.x * blockDim.x) + blockDim.x * blockIdx.x +
         threadIdx.x;
}

/// use this if you the 2D index (x,y) of a 2D array of size (nx,ny)
/// @note ny is not useful
inline __device__ void idx2d(int *x, int *y, const int nx) {
  int i1d = idx1d();
  *y = i1d / nx;
  *x = i1d - *y * nx;
}

/// use this if you the 3D index (x,y,z) of a 3D array of size (nx,ny,nz)
/// @note nz is not useful
inline __device__ void idx3d(int *x, int *y, int *z, const int nx,
                             const int ny) {
  int i1d = idx1d();
  *z = i1d / (nx * ny);
  int temp = i1d - *z * (nx * ny);
  *y = temp / nx;
  *x = temp - *y * nx;
}

struct CUDA_isNaN {
  CUDA_CALLABLE_MEMBER bool operator()(const real &a) const { return isnan(a); }
};

struct CUDA_isZero {
  CUDA_CALLABLE_MEMBER bool operator()(const real &a) const { return a == 0; }
};

/// check if there is any error and display the details if there are some
inline void CUDA_CHECK_ERRORS(const char *func_name) {
  cudaError_t cerror = cudaGetLastError();
  if (cerror != cudaSuccess) {
    char host[256];
    gethostname(host, 256);
    fprintf(stderr, "%s: CudaError: %s (on %s)\n", func_name,
            cudaGetErrorString(cerror), host);
    exit(1);
  }
}

#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(                                                                \
          stderr,                                                             \
          "CudaError: CUDA RT call \"%s\" in line %d of file %s failed with " \
          "%s (%d).\n",                                                       \
          #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),          \
          cudaStatus);                                                        \
  }

#define BLOCK_SIZE_DEFAULT 256

/** use this to compute the adequate grid dimensions when lauching a kernel
    create a 1D grid if nb_thread_total<nb_max_thread_per_block
    else create a 2D grid and dispatch the treads on it.

    @param[in]  nb_thread_total         number total of threads needed
    @param[in]  nb_max_thread_per_block number of thread per block you want
   (default: 256)
    @param[out] block_size              size of the blocs (3D)
    @param[out] grid_size               size of the grid  (3D)

    @note this is the runtime version
    \todo use unsigned int
    \todo make a debug mode to display the sizes
*/
// HACK to remove CU_SAFE_CALL_NO_SYNC
#define CU_SAFE_CALL_NO_SYNC(text) (text)
inline void setExtents(int nb_thread_total, int nb_thread_per_block,
                       dim3 *block_size, dim3 *grid_size) {
  int max_thread_per_block, nbtpb;
  CU_SAFE_CALL_NO_SYNC(cuDeviceGetAttribute(
      &max_thread_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0));
  if (nb_thread_per_block > max_thread_per_block) {
    nbtpb = max_thread_per_block;
    printf(
        "setExtents: WARNING, nb_thread_per_block is too big, using %d instead",
        max_thread_per_block);
  } else {
    nbtpb = nb_thread_per_block;
  }
  int nb_blocks_total =
      nb_thread_total / nbtpb + (nb_thread_total % nbtpb == 0 ? 0 : 1);
  int nb_blocks_x = nb_blocks_total;
  int nb_blocks_y = 1;

  int max_grid_dim_x;
  CU_SAFE_CALL_NO_SYNC(cuDeviceGetAttribute(
      &max_grid_dim_x, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 0));
  while (nb_blocks_x > max_grid_dim_x - 1) {
    nb_blocks_x /= 2;
    nb_blocks_y *= 2;
  }
  if (nb_blocks_x * nb_blocks_y * nbtpb < nb_thread_total) nb_blocks_x++;

  block_size->x = nbtpb;
  block_size->y = 1;
  block_size->z = 1;

  grid_size->x = nb_blocks_x;
  grid_size->y = nb_blocks_y;
  grid_size->z = 1;
  // cout << "Bloc : x, "<<block_size.x <<"; y, "<<block_size.y <<"; z,
  // "<<block_size.z << endl; cout << "Grid : x, "<<grid_size.x <<"; y,
  // "<<grid_size.y <<"; z, "<<grid_size.z << endl;
}
