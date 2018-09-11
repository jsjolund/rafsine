/**
  \file
      ArrayGPU.h
  \brief
      Helper class to define array on the GPU.
	  Uses thrust vectors as a basis.
  \author
      Nicolas Delbosc, University of Leeds
*/

#pragma once

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "CudaUtils.hpp"

//Define an group of array on the GPU
//designed for 3D arrays (but works in 2D as well)
//Useful to group all the distribution functions into a single array
//distribution functions (fi) are packed in memory based on their direction:
// memory: f1,f1,...,f1,f2,f2,...,f2,f3,f3,...
class DistributionFunctionsGroup
{
private:
  //number of arrays (or directions for distribution functions)
  const unsigned int m_Q;
  //number of nodes along each axis
  const unsigned int m_sizeX;
  const unsigned int m_sizeY;
  const unsigned int m_sizeZ;
  //distribution functions on the GPU
  thrust::device_vector<real> m_dfGPU;
  //true if the arrays are also allocated and usable on the CPU
  bool m_useCPU;
  //distribution functions on the CPU
  thrust::host_vector<real> m_dfCPU;

public:
  //Constructor
  DistributionFunctionsGroup(unsigned int Q,
                             unsigned int sizeX,
                             unsigned int sizeY,
                             unsigned int sizeZ = 1,
                             bool useCPU = true)
      : m_Q(Q),
        m_sizeX(sizeX),
        m_sizeY(sizeY),
        m_sizeZ(sizeZ),
        m_useCPU(useCPU),
        m_dfCPU(useCPU ? Q * sizeX * sizeY * sizeZ : 1),
        m_dfGPU(Q * sizeX * sizeY * sizeZ)
  {
  }

  //return the amount of memory used by the group of arrays
  //notes: same amount on both CPU and GPU
  inline unsigned long int memoryUse()
  {
    return sizeof(real) * m_Q * m_sizeX * m_sizeY * m_sizeZ;
  }

  //return the number of arrays in the group
  //i.e. the number of distribution functions
  inline unsigned int Q() { return m_Q; }

  //return size of the lattice
  inline unsigned int sizeX() { return m_sizeX; }
  inline unsigned int sizeY() { return m_sizeY; }
  inline unsigned int sizeZ() { return m_sizeZ; }
  inline unsigned int fullSize() { return m_sizeX * m_sizeY * m_sizeZ; }

  //1D access to distribution function on the CPU
  inline real &operator()(unsigned int df_idx, unsigned int idx)
  {
    if (m_useCPU)
      return m_dfCPU[idx + df_idx * m_sizeX * m_sizeY * m_sizeZ];
    else
    {
      std::cerr << "Error in 'DistributionFunctionsGroup::operator()'  CPU memory is not allocated." << std::endl;
      return m_dfCPU[0];
    }
  }
  //3D access to distribution function on the CPU
  inline real &operator()(unsigned int df_idx, unsigned int x, unsigned int y, unsigned int z = 0)
  {
    return (*this)(df_idx, x + y * m_sizeX + z * m_sizeX * m_sizeY);
  }

  //upload the distributions functions from the CPU to the GPU
  inline DistributionFunctionsGroup &upload()
  {
    if (m_useCPU)
      m_dfGPU = m_dfCPU;
    else
      std::cerr << "Error in 'DistributionFunctionsGroup::upload()'  CPU memory is not allocated." << std::endl;
    return *this;
  }
  //download the distributions functions from the GPU to the CPU
  inline DistributionFunctionsGroup &download()
  {
    if (m_useCPU)
      m_dfCPU = m_dfGPU;
    else
      std::cerr << "Error in 'DistributionFunctionsGroup::upload()'  CPU memory is not allocated." << std::endl;
    return *this;
  }
  //return a pointer to the beginning of the GPU memory
  inline real *gpu_ptr(unsigned int df_idx = 0)
  {
    const int SIZE = m_sizeX * m_sizeY * m_sizeZ;
    return thrust::raw_pointer_cast(&(m_dfGPU)[df_idx * SIZE]);
  }

  //copy from another group of distribution functions
  //SAME SIZE IS REQUIRED
  inline DistributionFunctionsGroup &operator=(const DistributionFunctionsGroup &f)
  {
    if ((m_Q == f.m_Q) && (m_sizeX == f.m_sizeX) && (m_sizeY == f.m_sizeY) && (m_sizeZ == f.m_sizeZ))
    {
      if (m_useCPU)
        thrust::copy(f.m_dfCPU.begin(), f.m_dfCPU.end(), m_dfCPU.begin());
      thrust::copy(f.m_dfGPU.begin(), f.m_dfGPU.end(), m_dfGPU.begin());
    }
    else
    {
      std::cerr << "Error in 'DistributionFunctionsGroup::operator ='  sizes do not match." << std::endl;
    }
    m_useCPU = f.m_useCPU;
    return *this;
  }

  //static function to swap two DistributionFunctionsGroup
  static inline void swap(DistributionFunctionsGroup &f1, DistributionFunctionsGroup &f2)
  {
    f1.m_dfCPU.swap(f2.m_dfCPU);
    f1.m_dfGPU.swap(f2.m_dfGPU);
  }

  //fill the ith array, i.e. the ith distribution function
  //with a constant value for all nodes
  inline void fill(unsigned int df_idx, real value)
  {
    const int SIZE = m_sizeX * m_sizeY * m_sizeZ;
    if (m_useCPU)
      thrust::fill(m_dfCPU.begin() + df_idx * SIZE, m_dfCPU.begin() + (df_idx + 1) * SIZE, value);
    thrust::fill(m_dfGPU.begin() + df_idx * SIZE, m_dfGPU.begin() + (df_idx + 1) * SIZE, value);
  }

  //return an iterator to the beggining of the ith distribution function (GPU)
  inline thrust::device_vector<real>::iterator begin(unsigned int df_idx)
  {
    const int SIZE = m_sizeX * m_sizeY * m_sizeZ;
    return m_dfGPU.begin() + df_idx * SIZE;
  }

  //return an iterator to the end of the ith distribution function (GPU)
  inline thrust::device_vector<real>::iterator end(unsigned int df_idx)
  {
    const int SIZE = m_sizeX * m_sizeY * m_sizeZ;
    return m_dfGPU.begin() + (df_idx + 1) * SIZE;
  }
  //Clear the arrays (free the memory)
  inline void clear()
  {
    m_dfCPU.clear();
    m_dfCPU.shrink_to_fit();
    m_dfGPU.clear();
    m_dfGPU.shrink_to_fit();
  }
};
