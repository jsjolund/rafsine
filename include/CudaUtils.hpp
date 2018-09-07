/** \file 
 *     cuda_utils.h
 *  \brief 
 *     Define several functions to ease the use of cuda kernels
 *  \details 
 *     Functions to check CUDA error list, 
 *     to set the dimension of the grid when launching kernels,
 *     to compute thread index in 1D, 2D, or 3D
 *  \author 
 *     Nicolas Delbosc
 *  \version 
 *     0.0a
 *  \date 
 *     november 2011
 **/
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda.h>
//#include <cutil_inline.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <thrust/device_vector.h>

//define how to index memory
#define I3D(x, y, z, nx, ny, nz) ((x) + (y) * (nx) + (z) * (nx) * (ny))
#define I4D(i, x, y, z, nx, ny, nz) ((i) * (nx) * (ny) * (nz) + (x) + (y) * (nx) + (z) * (nx) * (ny))
#define df3D(i, x, y, z, nx, ny, nz) (df[I4D(i, x, y, z, nx, ny, nz)])
#define dftmp3D(i, x, y, z, nx, ny, nz) (df_tmp[I4D(i, x, y, z, nx, ny, nz)])
#define Tdf3D(i, x, y, z, nx, ny, nz) (dfT[I4D(i, x, y, z, nx, ny, nz)])
#define Tdftmp3D(i, x, y, z, nx, ny, nz) (dfT_tmp[I4D(i, x, y, z, nx, ny, nz)])

/// Define the precision used for describing real number
typedef float real;
typedef float3 real3;

/// check if there is any error and display the details if there are some
inline void cuda_check_errors(const char *func_name)
{
  cudaError_t cerror = cudaGetLastError();
  if (cerror != cudaSuccess)
  {
    char host[256];
    gethostname(host, 256);
    printf("%s: CudaError: %s (on %s)\n", func_name, cudaGetErrorString(cerror), host);
    exit(1);
  }
}

#define BLOCK_SIZE_DEFAULT 256

/** use this to compute the adequate grid dimensions when lauching a kernel
    create a 1D grid if nb_thread_total<nb_max_thread_per_block
    else create a 2D grid and dispatch the treads on it.

    @param[in]  nb_thread_total         number total of threads needed
    @param[in]  nb_max_thread_per_block number of thread per block you want (default: 256)
    @param[out] block_size              size of the blocs (3D)
    @param[out] grid_size               size of the grid  (3D)

    @note this is the runtime version
    \todo use unsigned int 
    \todo make a debug mode to display the sizes
*/
//HACK to remove CU_SAFE_CALL_NO_SYNC
#define CU_SAFE_CALL_NO_SYNC(text) (text)
inline void setDims(int nb_thread_total, int nb_thread_per_block, dim3 &block_size, dim3 &grid_size)
{
  int max_thread_per_block, nbtpb;
  CU_SAFE_CALL_NO_SYNC(cuDeviceGetAttribute(&max_thread_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0));
  if (nb_thread_per_block > max_thread_per_block)
  {
    nbtpb = max_thread_per_block;
    printf("setDims: WARNING, nb_thread_per_block is too big, using %d instead", max_thread_per_block);
  }
  else
    nbtpb = nb_thread_per_block;

  int nb_blocks_total = nb_thread_total / nbtpb + (nb_thread_total % nbtpb == 0 ? 0 : 1);
  int nb_blocks_x = nb_blocks_total;
  int nb_blocks_y = 1;

  int max_grid_dim_x;
  CU_SAFE_CALL_NO_SYNC(cuDeviceGetAttribute(&max_grid_dim_x, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 0));
  while (nb_blocks_x > max_grid_dim_x - 1)
  {
    nb_blocks_x /= 2;
    nb_blocks_y *= 2;
  }
  if (nb_blocks_x * nb_blocks_y * nbtpb < nb_thread_total)
    nb_blocks_x++;

  block_size.x = nbtpb;
  block_size.y = 1;
  block_size.z = 1;

  grid_size.x = nb_blocks_x;
  grid_size.y = nb_blocks_y;
  grid_size.z = 1;
  //cout << "Bloc : x, "<<block_size.x <<"; y, "<<block_size.y <<"; z, "<<block_size.z << endl;
  //cout << "Grid : x, "<<grid_size.x <<"; y, "<<grid_size.y <<"; z, "<<grid_size.z << endl;
}

/** Template version of setDims()
 *  nb_thread_total and nb_thread_per_block needs to be compile-time constant
 *  but block_size and grid_size should be computed at compile time
 *  TODO use unsigned int 
 */

/// The following functions are used inside the kernel in order to access the local index of each thread

/// use this if you want a 1D index
inline __device__ int idx1d(void)
{
  return blockIdx.y * (gridDim.x * blockDim.x) + blockDim.x * blockIdx.x + threadIdx.x;
}

/// use this if you the 2D index (x,y) of a 2D array of size (nx,ny)
/// @note ny is not useful
inline __device__ void idx2d(int &x, int &y, const int nx)
{
  int i1d = idx1d();
  y = i1d / nx;
  x = i1d - y * nx;
}

/// use this if you the 3D index (x,y,z) of a 3D array of size (nx,ny,nz)
/// @note nz is not useful
inline __device__ void idx3d(int &x, int &y, int &z, const int nx, const int ny)
{
  int i1d = idx1d();
  z = i1d / (nx * ny);
  int temp = i1d - z * (nx * ny);
  y = temp / nx;
  x = temp - y * nx;
}

#endif
