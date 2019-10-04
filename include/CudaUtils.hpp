#pragma once

#include <cuda.h>
#include <math.h>
#include <math_constants.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <unistd.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <glm/vec3.hpp>

#include "CudaMathHelper.h"

#define NaN std::numeric_limits<real>::quiet_NaN()

inline void hash_combine(std::size_t *seed) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t *seed, const T &v, Rest... rest) {
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  hash_combine(seed, rest...);
}

namespace std {
template <>
struct hash<glm::ivec3> {
  std::size_t operator()(const glm::ivec3 &p) const {
    using std::hash;
    std::size_t seed = 0;
    ::hash_combine(&seed, p.x, p.y, p.z);
    return seed;
  }
};
}  // namespace std

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

/// Compute the absolute value of a
template <class T>
inline T abs(const T &a) {
  return (a > 0) ? a : (-a);
}

/// Compute the minimum of a and b
template <class T>
inline const T &min(const T &a, const T &b) {
  return (a < b) ? a : b;
}

/// Compute the maximum of a and b
template <class T>
inline const T &max(const T &a, const T &b) {
  return (a > b) ? a : b;
}

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

// Define how to index memory
#define I3D(x, y, z, nx, ny, nz) ((x) + (y) * (nx) + (z) * (nx) * (ny))
#define I4D(i, x, y, z, nx, ny, nz) \
  ((i) * (nx) * (ny) * (nz) + (x) + (y) * (nx) + (z) * (nx) * (ny))
#define df3D(i, x, y, z, nx, ny, nz) (df[I4D(i, x, y, z, nx, ny, nz)])
#define dftmp3D(i, x, y, z, nx, ny, nz) (df_tmp[I4D(i, x, y, z, nx, ny, nz)])
#define Tdf3D(i, x, y, z, nx, ny, nz) (dfT[I4D(i, x, y, z, nx, ny, nz)])
#define Tdftmp3D(i, x, y, z, nx, ny, nz) (dfT_tmp[I4D(i, x, y, z, nx, ny, nz)])

/// Define the precision used for describing real number
typedef float real;
typedef float3 real3;

#define make_real3 make_float3
#define REAL_NAN CUDART_NAN_F
#define REAL_MAX FLT_MAX
#define REAL_MIN FLT_MIN

template <class T>
std::ostream &operator<<(std::ostream &os, thrust::device_vector<T> v) {
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<float>(os, ", "));
  return os;
}

inline std::ostream &operator<<(std::ostream &os, glm::ivec3 v) {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, glm::vec3 v) {
  os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return os;
}

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
  __host__ __device__ bool operator()(const real &a) const { return isnan(a); }
};

struct CUDA_isZero {
  __host__ __device__ bool operator()(const real &a) const { return a == 0; }
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
